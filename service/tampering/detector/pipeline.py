"""Pipeline orchestrator: file -> List[PageReport], one report per page.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from .decision import decide
from .detectors import DocTamperDetector, MVSSNetDetector
from .engine import MVSSNetEngine, TamperingEngine, build_mvssnet_engine
from .loader import Page, load
from .localizers import FaceLocalizer, build_face_localizer
from .report import (
    Artifacts,
    DocumentType,
    ExecutionMetadata,
    FaceDetection,
    PageReport,
)
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
    mvssnet_engine: Optional[MVSSNetEngine] = None,
    enable_mvssnet: bool = True,
) -> List[PageReport]:
    """Run the full tampering pipeline on every page of the given file."""
    pages = load(file_path)
    if not pages:
        return []

    localizer = _resolve_localizer(face_localizer, enable_face_localizer)
    doctamper = DocTamperDetector(engine)
    mvssnet = MVSSNetDetector(_resolve_mvssnet_engine(mvssnet_engine, enable_mvssnet))

    reports: List[PageReport] = []
    for page in pages:
        reports.append(_analyze_page(
            page=page,
            page_count=len(pages),
            localizer=localizer,
            doctamper=doctamper,
            mvssnet=mvssnet,
            thresholds=thresholds,
            document_type=document_type,
            output_dir=output_dir,
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


def _resolve_mvssnet_engine(
    provided: Optional[MVSSNetEngine], enable: bool,
) -> Optional[MVSSNetEngine]:
    if not enable:
        return None
    if provided is not None:
        return provided
    return build_mvssnet_engine()


def _analyze_page(
    page: Page,
    page_count: int,
    localizer: Optional[FaceLocalizer],
    doctamper: DocTamperDetector,
    mvssnet: MVSSNetDetector,
    thresholds: Thresholds,
    document_type: DocumentType,
    output_dir: Optional[str],
) -> PageReport:
    timings: dict[str, int] = {}

    face = _run_face_localizer(page, localizer, document_type, timings)

    t0 = time.perf_counter()
    doctamper_result, doctamper_heatmap = doctamper.analyze(
        page.image, face, thresholds,
    )
    timings["doctamper"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    mvssnet_result, mvssnet_heatmap = mvssnet.analyze(page.image, face)
    timings["mvssnet"] = int((time.perf_counter() - t0) * 1000)

    verdict, verdict_score, reasons, confidence = decide(
        doctamper_result, mvssnet_result, face, thresholds,
    )

    artifacts = _persist_artifacts(
        page=page,
        doctamper_heatmap=doctamper_heatmap,
        doctamper_regions=doctamper_result.regions,
        mvssnet_heatmap=mvssnet_heatmap,
        face=face,
        verdict=verdict,
        verdict_score=verdict_score,
        output_dir=output_dir,
    )

    timings["total"] = sum(timings.values())

    metadata = ExecutionMetadata(
        document_type=document_type,
        page_index=page.page_number,
        page_count=page_count,
        image_shape=(page.image.shape[0], page.image.shape[1]),
        execution_ms=timings,
        thresholds_used=thresholds.as_dict(),
        model_versions={
            "doctamper": doctamper.engine_name,
            "face_localizer": localizer.name if localizer is not None else "disabled",
            "mvssnet": mvssnet.name,
        },
    )

    return PageReport(
        source=page.source,
        verdict=verdict,
        verdict_score=round(float(verdict_score), 3),
        verdict_reasons=reasons,
        confidence=confidence,
        face=face,
        doctamper=doctamper_result,
        mvssnet=mvssnet_result,
        artifacts=artifacts,
        metadata=metadata,
    )


def _run_face_localizer(
    page: Page,
    localizer: Optional[FaceLocalizer],
    document_type: DocumentType,
    timings: dict,
) -> FaceDetection:
    if localizer is None:
        return FaceDetection(detected=False, expected=True)
    t0 = time.perf_counter()
    try:
        face = localizer.locate(page.image, document_type=document_type)
    except Exception as exc:
        print(f"[WARN] Face localizer raised {exc.__class__.__name__}: {exc}")
        face = FaceDetection(detected=False, expected=True)
    timings["face"] = int((time.perf_counter() - t0) * 1000)
    return face


def _persist_artifacts(
    page: Page,
    doctamper_heatmap,
    doctamper_regions,
    mvssnet_heatmap,
    face: FaceDetection,
    verdict,
    verdict_score: float,
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
        verdict=verdict,
        verdict_score=verdict_score,
    )
    face_crop_path = save_face_crop(
        page.image, face, out / f"{stem}_face_crop.png",
    )

    mvssnet_heatmap_path = None
    if mvssnet_heatmap is not None:
        mvssnet_heatmap_path = save_heatmap(
            mvssnet_heatmap, out / f"{stem}_mvssnet_heatmap.png",
        )

    return Artifacts(
        source_image=None,
        doctamper_heatmap=str(heatmap_path),
        doctamper_overlay=str(overlay_path),
        mvssnet_heatmap=str(mvssnet_heatmap_path) if mvssnet_heatmap_path else None,
        face_crop=str(face_crop_path) if face_crop_path else None,
        combined_overlay=str(overlay_path),
    )
