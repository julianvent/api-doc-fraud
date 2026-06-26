from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedElement:
    # Coordinate space: absolute pixels, same polygon format as TextLine.bbox.
    # Persisted bboxes (FieldSpec.bbox) are normalized 0–1; conversion happens at Step 6.
    id: str           # stable, e.g. "el_0" — assigned at detection, never a list index
    text: str
    bbox: np.ndarray  # polygon of points, shape (N, 2)
    confidence: float
