from __future__ import annotations

from typing import List

import numpy as np

from typing import Any

from ...models import Word

# OCREngine here means the EasyOCR Reader interface (readtext returns a list).
# We accept Any to keep this internal helper decoupled from the public OCREngine
# contract used by the orchestrator (which exposes extract/warmup/close).
OCREngine = Any


def detect(image: np.ndarray, engine: OCREngine) -> List[Word]:
    """Run the OCR engine on an image and return normalized Word detections.

    EasyOCR returns (bbox, text, confidence) tuples where bbox is four corner
    points. We collapse to an axis-aligned rect and normalize to [0, 1].
    """
    h, w = image.shape[:2]
    raw = engine.readtext(image)

    words: List[Word] = []
    for bbox, text, confidence in raw:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        geometry = (
            (round(x1 / w, 4), round(y1 / h, 4)),
            (round(x2 / w, 4), round(y2 / h, 4)),
        )
        words.append(Word(
            text=text,
            confidence=round(float(confidence), 3),
            geometry=geometry,
        ))
    return words
