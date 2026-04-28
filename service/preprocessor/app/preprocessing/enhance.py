from __future__ import annotations

import cv2
import numpy as np

from ..config import EnhanceConfig


def enhance(image: np.ndarray, cfg: EnhanceConfig) -> np.ndarray:
    """Recover borderline images via local contrast + unsharp masking.

    Applied only when the first quality pass fails, so we don't distort
    already-clean scans.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    equalized = clahe.apply(image)
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=cfg.unsharp_sigma)
    return cv2.addWeighted(equalized, cfg.unsharp_amount, blurred, cfg.unsharp_base, 0)
