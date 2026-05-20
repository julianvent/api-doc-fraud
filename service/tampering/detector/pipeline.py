"""Pipeline orchestrator: file -> List[PageReport], one report per page.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detectors import DocTamperDetector, TruForDetector
from .engine import (
    TamperingEngine,
    TruForEngine,
    build_trufor_engine,
)
from .loader import Page, load
from .localizers import FaceLocalizer, build_face_localizer
from .report import (
    Artifacts,
    DocTamperResult,
    DocumentLocation,
    DocumentType,
    ExecutionMetadata,
    FaceDetection,
    PageReport,
    Region,
    Timings,
    TruForResult,
)
from .scoring import score
from .thresholds import DEFAULT, Thresholds
from .visualization import save_face_crop, save_heatmap, save_overlay


def analyze(
    file_path: str,
    engine: TamperingEngine,
    thresholds: Thresholds = DEFAULT,
    output_dir: Optional[str] = None,
    document_type: DocumentType = DocumentType.UNKNOWN,
    face_localizer: Optional[FaceLocalizer] = None,
    enable_face_localizer: bool = True,
    trufor_engine: Optional[TruForEngine] = None,
    enable_trufor: bool = True,
    page_locations: Optional[Dict[int, DocumentLocation]] = None,
) -> List[PageReport]:
    """Run the pipeline on every page. `page_locations` injects pre-computed
    DocAligner crops; detectors then run on the crop and results are
    remapped back to the original frame."""
    pages = load(file_path)
    if not pages:
        return []

    localizer = _resolve_localizer(face_localizer, enable_face_localizer)
    doctamper = DocTamperDetector(engine)
    trufor = TruForDetector(_resolve_trufor_engine(trufor_engine, enable_trufor))

    reports: List[PageReport] = []
    for page in pages:
        location = (page_locations or {}).get(page.page_number)
        reports.append(_analyze_page(
            page=page,
            page_count=len(pages),
            localizer=localizer,
            doctamper=doctamper,
            trufor=trufor,
            thresholds=thresholds,
            document_type=document_type,
            output_dir=output_dir,
            location=location,
        ))
    return reports


def _resolve_localizer(
    provided: Optional[FaceLocalizer], enable: bool,
) -> Optional[FaceLocalizer]:
    if not enable:
        return None
    if provided is not None:
        return provided
    try:
        return build_face_localizer()
    except RuntimeError as exc:
        print(f"[WARN] Face localizer unavailable: {exc}")
        return None


def _resolve_trufor_engine(
    provided: Optional[TruForEngine], enable: bool,
) -> Optional[TruForEngine]:
    if not enable:
        return None
    if provided is not None:
        return provided
    return build_trufor_engine()


def _analyze_page(
    page: Page,
    page_count: int,
    localizer: Optional[FaceLocalizer],
    doctamper: DocTamperDetector,
    trufor: TruForDetector,
    thresholds: Thresholds,
    document_type: DocumentType,
    output_dir: Optional[str],
    location: Optional[DocumentLocation] = None,
) -> PageReport:
    timings: dict[str, int] = {}

    frame_h, frame_w = page.image.shape[:2]
    if location is None:
        location = DocumentLocation(localized=False, frame_shape=(frame_h, frame_w))

    use_crop = location.localized and location.bbox_with_padding is not None
    if use_crop:
        cx, cy, cw, ch = location.bbox_with_padding
        model_image = page.image[cy:cy + ch, cx:cx + cw].copy()
    else:
        cx, cy, cw, ch = 0, 0, frame_w, frame_h
        model_image = page.image

    face_in_model = _run_face_localizer(model_image, localizer, document_type, timings)

    t0 = time.perf_counter()
    doctamper_result, doctamper_heatmap = doctamper.analyze(
        model_image, face_in_model, thresholds,
    )
    timings["doctamper"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    trufor_result, trufor_heatmap = trufor.analyze(model_image)
    timings["trufor"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    face_trufor_result = _run_trufor_on_face(page.image, face_in_model, (cx, cy), trufor)
    timings["face_trufor"] = int((time.perf_counter() - t0) * 1000)

    if use_crop:
        doctamper_result = _remap_doctamper_result(
            doctamper_result, (cx, cy), (cw, ch), (frame_h, frame_w),
        )
        face = _remap_face_to_frame(face_in_model, (cx, cy))
        doctamper_heatmap_for_overlay = _embed_heatmap_in_frame(
            doctamper_heatmap, (cx, cy), (frame_h, frame_w),
        )
    else:
        face = face_in_model
        doctamper_heatmap_for_overlay = doctamper_heatmap

    fraud_score, risk_label, reliability, findings = score(
        doctamper=doctamper_result,
        trufor=trufor_result,
        face=face,
        face_trufor=face_trufor_result,
        thresholds=thresholds,
    )

    artifacts = _persist_artifacts(
        page=page,
        doctamper_heatmap=doctamper_heatmap_for_overlay,
        doctamper_regions=doctamper_result.regions,
        trufor_heatmap=trufor_heatmap,
        face=face,
        risk_label=risk_label,
        fraud_score=fraud_score,
        output_dir=output_dir,
    )

    total_ms = sum(timings.values())
    timings_obj = Timings(total_ms=total_ms, per_detector=dict(timings))

    metadata = ExecutionMetadata(
        document_type=document_type,
        page_index=page.page_number,
        page_count=page_count,
        image_shape=(frame_h, frame_w),
        thresholds_used=thresholds.as_dict(),
        model_versions={
            "doctamper": doctamper.engine_name,
            "face_localizer": localizer.name if localizer is not None else "disabled",
            "trufor": trufor.name,
            "document_localizer": "docaligner-onnx" if use_crop else "disabled",
        },
    )

    return PageReport(
        source=page.source,
        fraud_score=fraud_score,
        risk_label=risk_label,
        reliability=reliability,
        findings=findings,
        timings=timings_obj,
        face=face,
        doctamper=doctamper_result,
        trufor=trufor_result,
        face_trufor=face_trufor_result,
        artifacts=artifacts,
        metadata=metadata,
        document_location=location,
    )


def _run_face_localizer(
    image: np.ndarray,
    localizer: Optional[FaceLocalizer],
    document_type: DocumentType,
    timings: dict,
) -> FaceDetection:
    if localizer is None:
        return FaceDetection(detected=False, expected=True)
    t0 = time.perf_counter()
    try:
        face = localizer.locate(image, document_type=document_type)
    except Exception as exc:
        print(f"[WARN] Face localizer raised {exc.__class__.__name__}: {exc}")
        face = FaceDetection(detected=False, expected=True)
    timings["face"] = int((time.perf_counter() - t0) * 1000)
    return face


_FACE_CROP_PADDING_RATIO = 0.20


def _run_trufor_on_face(
    frame_image: np.ndarray,
    face_in_model: FaceDetection,
    model_origin: Tuple[int, int],
    trufor: TruForDetector,
) -> TruForResult:
    """Second TruFor pass on the face crop (20% padding) at full frame
    resolution. Returns `ran=False` if no face or crop too small."""
    if not face_in_model.detected or face_in_model.bbox is None:
        return TruForResult(ran=False, skip_reason="no face detected")

    cx_origin, cy_origin = model_origin
    fx, fy, fw, fh = face_in_model.bbox
    fx_frame = fx + cx_origin
    fy_frame = fy + cy_origin

    pad_x = int(fw * _FACE_CROP_PADDING_RATIO)
    pad_y = int(fh * _FACE_CROP_PADDING_RATIO)
    frame_h, frame_w = frame_image.shape[:2]
    x1 = max(0, fx_frame - pad_x)
    y1 = max(0, fy_frame - pad_y)
    x2 = min(frame_w, fx_frame + fw + pad_x)
    y2 = min(frame_h, fy_frame + fh + pad_y)
    if x2 <= x1 or y2 <= y1:
        return TruForResult(ran=False, skip_reason="invalid face crop bounds")

    face_crop = frame_image[y1:y2, x1:x2]
    result, _ = trufor.analyze(face_crop)
    return result


def _remap_doctamper_result(
    result: DocTamperResult,
    crop_origin: Tuple[int, int],
    crop_shape: Tuple[int, int],
    frame_shape: Tuple[int, int],
) -> DocTamperResult:
    """Remap region bboxes from crop coords to frame coords.

    `area_fraction` stays relative to the doc — thresholds are calibrated
    against doc area, not frame area.
    """
    cx, cy = crop_origin
    cw, ch = crop_shape
    fh, fw = frame_shape

    new_regions: List[Region] = []
    for r in result.regions:
        rx1, ry1, rx2, ry2 = r.bbox
        new_bbox = (
            round((cx + rx1 * cw) / fw, 4),
            round((cy + ry1 * ch) / fh, 4),
            round((cx + rx2 * cw) / fw, 4),
            round((cy + ry2 * ch) / fh, 4),
        )
        new_regions.append(Region(
            detector=r.detector,
            bbox=new_bbox,
            area_fraction=r.area_fraction,
            score=r.score,
            zone=r.zone,
            label=r.label,
        ))

    return DocTamperResult(
        ran=result.ran,
        skip_reason=result.skip_reason,
        score_mean=result.score_mean,
        score_max=result.score_max,
        score_outside_face=result.score_outside_face,
        regions=new_regions,
    )


def _remap_face_to_frame(
    face: FaceDetection, crop_origin: Tuple[int, int],
) -> FaceDetection:
    if not face.detected or face.bbox is None:
        return face
    cx, cy = crop_origin
    fx, fy, fw, fh = face.bbox
    return FaceDetection(
        detected=face.detected,
        bbox=(fx + cx, fy + cy, fw, fh),
        confidence=face.confidence,
        expected=face.expected,
    )


def _embed_heatmap_in_frame(
    heatmap: np.ndarray,
    crop_origin: Tuple[int, int],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Embed the crop-sized heatmap into a frame-sized zero canvas so the
    overlay can blend onto the full frame."""
    fh, fw = frame_shape
    canvas = np.zeros((fh, fw), dtype=heatmap.dtype)
    cx, cy = crop_origin
    ch, cw = heatmap.shape[:2]
    canvas[cy:cy + ch, cx:cx + cw] = heatmap
    return canvas


def _persist_artifacts(
    page: Page,
    doctamper_heatmap,
    doctamper_regions,
    trufor_heatmap,
    face: FaceDetection,
    risk_label,
    fraud_score: float,
    output_dir: Optional[str],
) -> Artifacts:
    if output_dir is None:
        return Artifacts()

    out = Path(output_dir)
    stem = f"{Path(page.source).stem}_p{page.page_number}"

    heatmap_path = save_heatmap(
        doctamper_heatmap, out / f"{stem}_doctamper_heatmap.png",
    )
    overlay_path = save_overlay(
        page.image,
        doctamper_heatmap,
        doctamper_regions,
        out / f"{stem}_overlay.png",
        face=face,
        risk_label=risk_label,
        fraud_score=fraud_score,
    )
    face_crop_path = save_face_crop(
        page.image, face, out / f"{stem}_face_crop.png",
    )

    trufor_heatmap_path = None
    if trufor_heatmap is not None:
        trufor_heatmap_path = save_heatmap(
            trufor_heatmap, out / f"{stem}_trufor_heatmap.png",
        )

    return Artifacts(
        source_image=None,
        doctamper_heatmap=str(heatmap_path),
        doctamper_overlay=str(overlay_path),
        trufor_heatmap=str(trufor_heatmap_path) if trufor_heatmap_path else None,
        face_crop=str(face_crop_path) if face_crop_path else None,
        combined_overlay=str(overlay_path),
    )
