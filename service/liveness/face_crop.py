"""Face cropping with padding — pure NumPy operations."""

from __future__ import annotations

import numpy as np

from service.liveness.domain.face import BoundingBox


def crop_with_padding(
    image: np.ndarray,
    bbox: BoundingBox,
    padding: float = 0.20,
) -> tuple[np.ndarray, BoundingBox]:
    """Crop around bbox expanded by `padding` (fraction of bbox size).

    Returns (crop, clamped_bbox); bbox is clamped to image bounds so the
    crop matches the reported rectangle. Padding 0.20 was validated on
    FaceTruFor in the tampering module (feedback_tampering_detectors).
    """
    if padding < 0.0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    h, w = image.shape[:2]
    pad_x = int(round(bbox.width * padding))
    pad_y = int(round(bbox.height * padding))

    x0 = max(0, bbox.x - pad_x)
    y0 = max(0, bbox.y - pad_y)
    x1 = min(w, bbox.x + bbox.width + pad_x)
    y1 = min(h, bbox.y + bbox.height + pad_y)

    clamped = BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    return image[y0:y1, x0:x1].copy(), clamped
