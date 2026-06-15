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


def _detect_faces(image: np.ndarray):
    cascade = _get_cascade()
    if cascade is None:
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    except Exception:
        return None


def has_face(image: np.ndarray) -> bool:
    faces = _detect_faces(image)
    return faces is not None and len(faces) > 0


def largest_face_area_ratio(image: np.ndarray) -> float:
    faces = _detect_faces(image)
    if faces is None or len(faces) == 0 or image.ndim < 2:
        return 0.0
    h, w  = image.shape[:2]
    total = h * w
    if total <= 0:
        return 0.0
    largest = max(fw * fh for (_, _, fw, fh) in faces)
    return largest / total
