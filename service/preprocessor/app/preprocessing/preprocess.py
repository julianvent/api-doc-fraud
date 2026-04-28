from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessingConfig
from ..models import Page


def preprocess(page: Page, cfg: PreprocessingConfig) -> Page:
    """Normalize a page for OCR: grayscale, resize, deskew, border.

    Deliberately minimal — only transformations that provably help OCR
    without destroying information.
    """
    image = _to_grayscale(page.image)
    image = _ensure_min_size(image, cfg.min_dimension)
    image = _deskew(image, cfg.min_deskew_angle, cfg.max_deskew_angle)
    image = _add_border(image, cfg.border_px)
    return Page(image=image, page_number=page.page_number, dpi=page.dpi, source=page.source)


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


def _deskew(image: np.ndarray, min_angle: float, max_angle: float) -> np.ndarray:
    """Correct small skew angles using Hough line detection."""
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=200, minLineLength=100, maxLineGap=10,
    )
    if lines is None:
        return image

    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in lines]
    angle = float(np.median(angles))

    if abs(angle) < min_angle or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _add_border(image: np.ndarray, border_px: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, border_px, border_px, border_px, border_px,
        borderType=cv2.BORDER_CONSTANT, value=255,
    )
