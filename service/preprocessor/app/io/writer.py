from __future__ import annotations

from pathlib import Path

import cv2

from ..models import ProcessedPage


def save_image(page: ProcessedPage, output_dir: str | Path) -> Path:
    """Persist the preprocessed page as a PNG. Returns the output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(page.source).stem
    path = out_dir / f"{src_name}_p{page.page_number}.png"
    cv2.imwrite(str(path), page.image)
    return path
