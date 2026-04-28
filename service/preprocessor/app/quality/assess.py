from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityConfig
from ..models import Page, QualityReport


def assess(page: Page, cfg: QualityConfig) -> QualityReport:
    """Evaluate whether a preprocessed page is suitable for OCR.

    Metrics are measured over detected text regions instead of the whole image,
    so large white margins or solid photo regions no longer drag the score down.
    """
    mask, heights = _detect_text_regions(
        page.image, cfg.adaptive_block_size, cfg.adaptive_c,
    )
    n_components = len(heights)

    if n_components < cfg.min_text_components:
        return QualityReport(
            text_sharpness=0.0, text_contrast=0.0, x_height_px=0.0,
            score=0.0, passed=False, rescued=False,
            reason=f"No text detected ({n_components} components)",
        )

    text_sharpness = _mean_gradient(page.image, mask)
    text_contrast = float(page.image[mask > 0].std())
    x_height_px = float(np.median(heights))

    s_norm = min(text_sharpness / cfg.target_text_sharpness, 1.0)
    c_norm = min(text_contrast / cfg.target_text_contrast, 1.0)
    x_norm = min(x_height_px / cfg.target_x_height_px, 1.0)
    score = (
        cfg.weight_sharpness * s_norm
        + cfg.weight_contrast * c_norm
        + cfg.weight_x_height * x_norm
    )

    passed = score >= cfg.threshold
    reason = "OK" if passed else f"Score {score:.3f} < {cfg.threshold}"
    return QualityReport(
        text_sharpness=text_sharpness,
        text_contrast=text_contrast,
        x_height_px=x_height_px,
        score=score, passed=passed, rescued=False, reason=reason,
    )


def _detect_text_regions(
    image: np.ndarray, block_size: int, c: int,
) -> tuple[np.ndarray, list[int]]:
    """Return a binary mask of text-like components and their heights."""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=block_size, C=c,
    )
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h, w = image.shape
    max_h = int(h * 0.10)
    max_w = int(w * 0.50)
    mask = np.zeros_like(image, dtype=np.uint8)
    heights: list[int] = []

    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if ch < 8 or ch > max_h:
            continue
        if cw < 2 or cw > max_w:
            continue
        if area < 15 or area > 5000:
            continue
        if cw / ch > 15:
            continue
        heights.append(int(ch))
        mask[y:y + ch, x:x + cw] = 255

    return mask, heights


def _mean_gradient(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude restricted to the mask."""
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag[mask > 0].mean())
