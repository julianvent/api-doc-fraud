from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# Normalized geometry: ((x1, y1), (x2, y2)) in [0, 1] coordinates.
Geometry = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class Word:
    """A single OCR detection with normalized geometry."""

    text: str
    confidence: float
    geometry: Geometry
    original: Optional[str] = None  # set when split from a mixed token


@dataclass
class ScriptBucket:
    """Per-script OCR output: reconstructed text + constituent words."""

    text: str
    words: List[Word]


@dataclass
class OCRResult:
    """Bilingual OCR output — contract consumed by the downstream agent.

    The shape of this dataclass IS the agent contract. Any change here must
    be coordinated with the agent.
    """

    english: ScriptBucket
    hindi: ScriptBucket
    low_confidence: List[Word]
