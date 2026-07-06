from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from service.ocr.models import TextLine


@dataclass
class DetectedElement:
    # Precondición: bbox debe estar normalizado 0–1 (PaddleOCR ya lo garantiza).
    # DotsOCR/Dolphin devuelven píxeles absolutos y están fuera del alcance de Step 1.
    # Persisted bboxes (FieldSpec.bbox) usan el mismo espacio 0–1; conversión en Step 6.
    id: int           # stable integer, assigned at detection
    text: str
    bbox: np.ndarray  # polygon of points, shape (N, 2), values in [0, 1]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "text":       self.text,
            "bbox":       self.bbox.tolist(),
            "confidence": float(self.confidence),
        }


def textlines_to_elements(lines: list[TextLine]) -> list[DetectedElement]:
    """Wrap engine.extract() output as DetectedElements with stable string IDs.

    Precondición: cada TextLine.bbox debe estar normalizado 0–1 (PaddleOCR lo garantiza).
    DotsOCR y Dolphin devuelven píxeles absolutos — están fuera del alcance de este step.
    Si se pasa un bbox no normalizado, la función lanza ValueError.

    IDs son deterministas dentro de la llamada: el_0, el_1, …
    Nada se agrupa ni se empareja.
    """
    elements = []
    for i, line in enumerate(lines):
        bbox = line.bbox.copy()
        if bbox.size > 0 and float(bbox.max()) > 1.5:
            raise ValueError(
                f"TextLine {i} has non-normalized bbox (max={bbox.max():.1f}). "
                "Expected values in [0, 1]. DotsOCR/Dolphin are out of scope for Step 1."
            )
        elements.append(DetectedElement(
            id=i,
            text=line.text,
            bbox=bbox,
            confidence=line.confidence,
        ))
    return elements
