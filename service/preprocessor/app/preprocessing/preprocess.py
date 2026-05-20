from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessingConfig
from ..models import Page
from .scan import scan


def preprocess(page: Page, cfg: PreprocessingConfig) -> Page:
    """Normalize a page for OCR: scan, grayscale, resize, border.

    The DocAligner-based scanner crops and dewarps the document when it finds
    a valid quadrilateral. When it does not, the original image is kept and
    only the OCR-friendly transformations (grayscale, upscale, border) apply.
    """
    page = scan(page, cfg.scan)
    image = _to_grayscale(page.image)
    image = _ensure_min_size(image, cfg.min_dimension)
    image = _add_border(image, cfg.border_px)
    return Page(
        image=image,
        page_number=page.page_number,
        dpi=page.dpi,
        source=page.source,
        scanned=page.scanned,
    )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ensure_min_size(image: np.ndarray, min_dimension: int) -> np.ndarray:
    """Upscale small images so text glyphs are large enough for OCR."""
    min_dim = min(image.shape[:2])
    if min_dim >= min_dimension:
        return image
    scale = min_dimension / min_dim
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _add_border(image: np.ndarray, border_px: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, border_px, border_px, border_px, border_px,
        borderType=cv2.BORDER_CONSTANT, value=255,
    )
