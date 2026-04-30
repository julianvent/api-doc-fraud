"""Facade for the OCR module. Consumed by the orchestrator.

The selected engine is built lazily and cached so EasyOCR's expensive Reader
is created at most once per process. Call `warmup()` from the FastAPI lifespan
to pay the load cost at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from service.ocr.app.config import Config, load as load_config
from service.ocr.app.contract import OCREngine
from service.ocr.app.factory import build_engine
from service.ocr.app.models import OCRResult
from service.ocr.app.visualization.mpl_overlay import render as render_overlay
from service.ocr.paths import OUTPUT_DIR
from service.preprocessor.app.models import ProcessedPage

DEFAULT_ENGINE = "easyocr"

_cfg: Optional[Config] = None
_engine: Optional[OCREngine] = None
_engine_name: Optional[str] = None


def _get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def warmup(engine_name: str = DEFAULT_ENGINE) -> None:
    """Eagerly build and load the selected engine."""
    global _engine, _engine_name
    cfg = _get_cfg()
    if _engine is None or _engine_name != engine_name:
        _engine = build_engine(engine_name, cfg.ocr)
        _engine_name = engine_name
    _engine.warmup()


def extract(
    pages: list[ProcessedPage],
    engine_name: str = DEFAULT_ENGINE,
    output_subdir: Optional[str] = None,
) -> List[OCRResult]:
    """Run OCR over every preprocessed page. One OCRResult per page.

    If `output_subdir` is given, an annotated PNG with bboxes per word is
    saved under `<service/ocr>/output/<output_subdir>/`.
    """
    warmup(engine_name)
    cfg = _get_cfg()
    assert _engine is not None

    out_dir: Optional[Path] = None
    if output_subdir is not None:
        out_dir = OUTPUT_DIR / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

    results: List[OCRResult] = []
    for p in pages:
        result = _engine.extract(p.image, cfg.ocr)
        results.append(result)
        if out_dir is not None:
            stem = Path(p.source).stem
            target = out_dir / f"{stem}_p{p.page_number}_ocr.png"
            render_overlay(p.image, result, target, cfg.visualization.font_path)
    return results
