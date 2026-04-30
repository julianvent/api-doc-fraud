"""Facade for the preprocessor module. Consumed by the orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from service.preprocessor.app.config import Config, load as load_config
from service.preprocessor.app.models import ProcessedPage
from service.preprocessor.app.service import process_file as _process_file

_cfg: Optional[Config] = None


def _get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def process(paths: list[Path | str]) -> List[ProcessedPage]:
    """Run preprocessor + quality assessment on every file. Flat page list."""
    cfg = _get_cfg()
    out: List[ProcessedPage] = []
    for p in paths:
        out.extend(_process_file(p, cfg))
    return out
