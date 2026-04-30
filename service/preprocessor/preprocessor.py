"""Facade for the preprocessor module. Consumed by the orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from service.preprocessor.app.config import Config, load as load_config
from service.preprocessor.app.io.writer import save_image
from service.preprocessor.app.models import ProcessedPage
from service.preprocessor.app.service import process_file as _process_file
from service.preprocessor.paths import OUTPUT_DIR

_cfg: Optional[Config] = None


def _get_cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def process(
    paths: list[Path | str],
    output_subdir: Optional[str] = None,
) -> List[ProcessedPage]:
    """Run preprocessor + quality assessment on every file. Flat page list.

    If `output_subdir` is given, the preprocessed PNG of each page is saved
    under `<service/preprocessor>/output/<output_subdir>/`.
    """
    cfg = _get_cfg()
    out_dir: Optional[Path] = None
    if output_subdir is not None:
        out_dir = OUTPUT_DIR / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

    out: List[ProcessedPage] = []
    for p in paths:
        pages = _process_file(p, cfg)
        if out_dir is not None:
            for page in pages:
                save_image(page, out_dir)
        out.extend(pages)
    return out
