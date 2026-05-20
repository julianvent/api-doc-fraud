"""TruFor general manipulation detector."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..engine import TruForEngine
from ..report import TruForResult

# Softmax decision boundary for binarizing the heatmap.
_SUSPICIOUS_PIXEL_THRESHOLD = 0.5
# Below this side, SegFormer's 32x downsampling collapses output.
_MIN_INPUT_SIDE = 128


class TruForDetector:
    name = "trufor-cmx-mit_b2"

    def __init__(self, engine: Optional[TruForEngine] = None) -> None:
        self._engine = engine

    def analyze(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[TruForResult, Optional[np.ndarray]]:
        if self._engine is None:
            return (
                TruForResult(ran=False, skip_reason="TruFor engine not loaded"),
                None,
            )

        if image_rgb is None or image_rgb.size == 0:
            return (
                TruForResult(ran=False, skip_reason="empty image"),
                None,
            )

        if min(image_rgb.shape[:2]) < _MIN_INPUT_SIDE:
            return (
                TruForResult(
                    ran=False,
                    skip_reason=f"image too small (<{_MIN_INPUT_SIDE}px side)",
                ),
                None,
            )

        heatmap, score = self._engine.detect(image_rgb)
        largest_area_fraction = _largest_suspicious_component_fraction(
            heatmap, _SUSPICIOUS_PIXEL_THRESHOLD,
        )
        result = TruForResult(
            ran=True,
            score=round(float(score), 4),
            largest_region_area_fraction=round(float(largest_area_fraction), 4),
        )
        return result, heatmap


def _largest_suspicious_component_fraction(
    heatmap: np.ndarray, pixel_threshold: float,
) -> float:
    """Fraction covered by the largest contiguous high-probability component.
    Real manipulations are contiguous; scanner noise is scattered."""
    if heatmap.size == 0:
        return 0.0
    binary = (heatmap >= pixel_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0
    largest_area = int(stats[1:, 4].max())
    return largest_area / float(heatmap.size)
