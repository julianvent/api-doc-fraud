from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from service.tampering.detector import (
    DocumentLocation,
    FaceLocalizer,
    PageReport,
    TamperingEngine,
    TruForEngine,
    analyze as _analyze,
    build_engine,
    build_face_localizer,
    build_trufor_engine,
)
from service.tampering.detector.loader import load
from service.tampering.document_crop import locate
from service.tampering.paths import OUTPUT_DIR

_engine: Optional[TamperingEngine] = None
_trufor_engine: Optional[TruForEngine] = None
_face_localizer: Optional[FaceLocalizer] = None
_warmed_up = False


def warmup() -> None:
    """Eagerly load all models. Safe to call multiple times."""
    global _engine, _trufor_engine, _face_localizer, _warmed_up
    if _warmed_up:
        return
    _engine = build_engine(model_name="auto", device="cpu")
    _trufor_engine = build_trufor_engine(device="cpu")
    try:
        _face_localizer = build_face_localizer()
    except Exception as exc:
        print(f"[WARN] Face localizer unavailable: {exc}")
        _face_localizer = None

    # Pay the DocAligner ONNX cold start once on a dummy input.
    try:
        locate(np.zeros((512, 512, 3), dtype=np.uint8))
    except Exception as exc:
        print(f"[WARN] DocAligner warmup skipped: {exc}")

    _warmed_up = True


def analyze(
    paths: list[Path | str],
    output_subdir: Optional[str] = None,
) -> List[PageReport]:
    """Run tampering analysis over every page. Returns a flat list. Artifacts
    are written under `<service/tampering>/output/<output_subdir>/` if set."""
    warmup()
    out_dir: Optional[Path] = None
    if output_subdir is not None:
        out_dir = OUTPUT_DIR / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[PageReport] = []
    for p in paths:
        locations = _build_locations(p)
        reports.extend(
            _analyze(
                file_path=str(p),
                engine=_engine,
                face_localizer=_face_localizer,
                enable_face_localizer=_face_localizer is not None,
                trufor_engine=_trufor_engine,
                enable_trufor=_trufor_engine is not None,
                output_dir=str(out_dir) if out_dir is not None else None,
                page_locations=locations,
            )
        )
    return reports


def _build_locations(path: Path | str) -> Dict[int, DocumentLocation]:
    """Run DocAligner per page → {page_number: DocumentLocation}. Failures
    return a `localized=False` location so callers process the full frame."""
    locations: Dict[int, DocumentLocation] = {}
    try:
        pages = load(path)
    except Exception as exc:
        print(f"[WARN] document_crop could not load {path}: {exc}")
        return locations

    for page in pages:
        try:
            loc = locate(page.image)
        except Exception as exc:
            print(
                f"[WARN] document_crop.locate failed for "
                f"{page.source} page {page.page_number}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            h, w = page.image.shape[:2]
            loc = DocumentLocation(
                localized=False,
                frame_shape=(h, w),
                validation_status="EXCEPTION",
                fallback_reason=f"{exc.__class__.__name__}: {exc}",
            )

        if not loc.localized:
            print(
                f"[INFO] page {page.page_number} processed full-frame "
                f"({loc.validation_status}: {loc.fallback_reason})"
            )

        locations[page.page_number] = loc
    return locations
