"""Facade for the tampering module. Consumed by the orchestrator.

Engines are heavy to construct, so they are cached at module level. The
orchestrator may also call `warmup()` from FastAPI's lifespan to pay the
load cost once at server startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from service.tampering.detector import (
    FaceLocalizer,
    MVSSNetEngine,
    PageReport,
    TamperingEngine,
    analyze as _analyze,
    build_engine,
    build_face_localizer,
    build_mvssnet_engine,
)

_engine: Optional[TamperingEngine] = None
_mvssnet_engine: Optional[MVSSNetEngine] = None
_face_localizer: Optional[FaceLocalizer] = None
_warmed_up = False


def warmup() -> None:
    """Eagerly load all models. Safe to call multiple times."""
    global _engine, _mvssnet_engine, _face_localizer, _warmed_up
    if _warmed_up:
        return
    _engine = build_engine(model_name="auto", device="cpu")
    _mvssnet_engine = build_mvssnet_engine(device="cpu")
    try:
        _face_localizer = build_face_localizer()
    except Exception as exc:
        print(f"[WARN] Face localizer unavailable: {exc}")
        _face_localizer = None
    _warmed_up = True


def analyze(
    paths: list[Path | str],
    output_dir: Path | str,
) -> List[PageReport]:
    """Run tampering analysis over every page of every file. Flat list.

    Heatmaps, overlay and face crop are written to `output_dir` for visual
    inspection.
    """
    warmup()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    reports: List[PageReport] = []
    for p in paths:
        reports.extend(
            _analyze(
                file_path=str(p),
                engine=_engine,
                face_localizer=_face_localizer,
                enable_face_localizer=_face_localizer is not None,
                mvssnet_engine=_mvssnet_engine,
                enable_mvssnet=_mvssnet_engine is not None,
                output_dir=str(out_path),
            )
        )
    return reports
