"""MVSS-Net photo-splicing detector, operating on the face crop."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..engine import MVSSNetEngine
from ..report import FaceDetection, MVSSNetResult

# Expand the face bbox by this small fraction before cropping. Tight crops
# avoid including the photo frame, ID number panel, or security features
# that trigger MVSS-Net on otherwise-clean documents. Enough padding remains
# to preserve hair / jawline context around the face itself.
_FACE_PAD_FRACTION = 0.50

# MVSS-Net was trained on inputs of ~512 px. Feeding smaller crops forces
# aggressive upscaling, which blurs exactly the splicing boundaries the
# model reads. Below this side length we skip rather than emit a noisy signal.
_MIN_CROP_SIDE = 96

# Pixel threshold used to binarize the heatmap when measuring contiguous
# suspicious area. 0.5 is the natural decision boundary of the sigmoid head.
_SUSPICIOUS_PIXEL_THRESHOLD = 0.5


class MVSSNetDetector:
    name = "mvss-casia"

    def __init__(self, engine: Optional[MVSSNetEngine] = None) -> None:
        self._engine = engine

    def analyze(
        self,
        image_rgb: np.ndarray,
        face: FaceDetection,
    ) -> Tuple[MVSSNetResult, Optional[np.ndarray]]:
        if self._engine is None:
            return (
                MVSSNetResult(ran=False, skip_reason="MVSS-Net engine not loaded"),
                None,
            )

        if not face.detected or face.bbox is None:
            return (
                MVSSNetResult(ran=False, skip_reason="no face detected on page"),
                None,
            )

        crop, _ = _expand_and_crop(image_rgb, face.bbox, _FACE_PAD_FRACTION)
        if crop is None or min(crop.shape[:2]) < _MIN_CROP_SIDE:
            return (
                MVSSNetResult(
                    ran=False,
                    skip_reason=f"face crop too small (<{_MIN_CROP_SIDE}px side)",
                ),
                None,
            )

        heatmap, score = self._engine.detect(crop)
        largest_area_fraction = _largest_suspicious_component_fraction(
            heatmap, _SUSPICIOUS_PIXEL_THRESHOLD,
        )
        result = MVSSNetResult(
            ran=True,
            score=round(float(score), 4),
            largest_region_area_fraction=round(float(largest_area_fraction), 4),
        )
        return result, heatmap


def _largest_suspicious_component_fraction(
    heatmap: np.ndarray, pixel_threshold: float,
) -> float:
    """Return the fraction of the heatmap covered by its largest connected
    high-probability component.

    A real spliced photo produces one contiguous hot region; scanner noise
    produces many small scattered hot pixels. Taking only the *largest*
    component separates those cases cleanly.
    """
    if heatmap.size == 0:
        return 0.0
    binary = (heatmap >= pixel_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0
    # stats[0] is the background; stats[i, 4] is the area of component i.
    largest_area = int(stats[1:, 4].max())
    return largest_area / float(heatmap.size)


def _expand_and_crop(
    image_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    pad_fraction: float,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Expand the face bbox by `pad_fraction` on all sides and crop.

    Returns (crop, padded_bbox) in original image coordinates, or (None, None)
    when the bbox collapses to zero area after clamping.
    """
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = face_bbox
    pad_x = int(round(bw * pad_fraction))
    pad_y = int(round(bh * pad_fraction))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return image_rgb[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)
