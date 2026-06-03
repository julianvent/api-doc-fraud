from typing import Optional

import cv2
import numpy as np


_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> Optional[cv2.CascadeClassifier]:
    global _CASCADE
    if _CASCADE is not None:
        return _CASCADE
    try:
        path    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None
        _CASCADE = cascade
        return _CASCADE
    except Exception:
        return None


def has_face(image: np.ndarray) -> bool:
    cascade = _get_cascade()
    if cascade is None:
        return False
    try:
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.1,
            minNeighbors = 5,
            minSize      = (40, 40),
        )
        return len(faces) > 0
    except Exception:
        return False
