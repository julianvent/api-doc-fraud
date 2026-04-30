"""Load image or PDF files into a uniform list of RGB pages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz
import numpy as np
from PIL import Image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_EXTS = {".pdf"}
_FITZ_BASE_DPI = 72.0


@dataclass
class Page:
    image: np.ndarray     
    page_number: int
    source: str


def load(file_path: str | Path, pdf_dpi: int = 200) -> List[Page]:
    """Load a file as a list of pages. PDFs are rasterized at `pdf_dpi`."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in _PDF_EXTS:
        return _load_pdf(path, pdf_dpi)
    if ext in _IMAGE_EXTS:
        return _load_image(path)
    raise ValueError(f"Unsupported format: {ext}")


def _load_image(path: Path) -> List[Page]:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return [Page(image=arr, page_number=1, source=str(path))]


def _load_pdf(path: Path, dpi: int) -> List[Page]:
    zoom = dpi / _FITZ_BASE_DPI
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(str(path))
    pages: List[Page] = []
    try:
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n).copy()
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            pages.append(Page(image=arr, page_number=i + 1, source=str(path)))
    finally:
        doc.close()
    return pages
