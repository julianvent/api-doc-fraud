"""Facade for the preprocessor module. Consumed by the orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from service.preprocessor.app.config import Config, load as load_config
from service.preprocessor.app.io.writer import save_image
from service.preprocessor.app.models import ProcessedPage
from service.preprocessor.app.service import process_file as _process_file

_cfg: Optional[Config] = None


def _get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def process(
    paths: list[Path | str],
    output_dir: Path | str,
) -> List[ProcessedPage]:
    """Run preprocessor + quality assessment on every file. Flat page list.

    Every preprocessed page is saved to `output_dir` as PNG for visual
    inspection.
    """
    cfg = _get_cfg()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    out: List[ProcessedPage] = []
    for p in paths:
        pages = _process_file(p, cfg)
        for page in pages:
            save_image(page, out_path)
        out.extend(pages)
    return out
