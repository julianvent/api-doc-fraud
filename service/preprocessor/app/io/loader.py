from __future__ import annotations

from pathlib import Path
from typing import List, Set

import cv2
import fitz
import numpy as np

from ..models import Page

SUPPORTED_IMAGES: Set[str] = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
}
SUPPORTED_ALL: Set[str] = SUPPORTED_IMAGES | {".pdf"}
_FITZ_BASE_DPI: float = 72.0


def load(file_path: str, dpi: int = 300) -> List[Page]:
    """Load a PDF or image file and return its pages as raster images."""
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_IMAGES:
        return _load_image(file_path, dpi)
    if ext == ".pdf":
        return _load_pdf(file_path, dpi)
    raise ValueError(f"Unsupported format: {ext}")


def _load_image(path: str, dpi: int) -> List[Page]:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return [Page(image=image, page_number=1, dpi=dpi, source=path)]


def _load_pdf(path: str, dpi: int) -> List[Page]:
    zoom = dpi / _FITZ_BASE_DPI
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(path)
    pages: List[Page] = []
    try:
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=matrix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n).copy()
            pages.append(Page(image=arr, page_number=i + 1, dpi=dpi, source=path))
    finally:
        doc.close()
    return pages
