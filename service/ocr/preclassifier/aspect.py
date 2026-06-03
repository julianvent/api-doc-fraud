from typing import Optional

import numpy as np


def classify(image: np.ndarray) -> Optional[str]:
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if h == 0:
        return None

    ratio = w / h
    if 1.55 <= ratio <= 1.62:
        return "TD1"
    if 0.61 <= ratio <= 0.66:
        return "TD1"
    return None
