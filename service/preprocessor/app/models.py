from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Page:
    """A single document page as a raster image."""

    image: np.ndarray
    page_number: int
    dpi: int
    source: str


@dataclass
class QualityReport:
    """Quality assessment result for a single page."""

    text_sharpness: float
    text_contrast: float
    x_height_px: float
    score: float
    passed: bool
    reason: str
    rescued: bool = False


@dataclass
class ProcessedPage:
    """A page after preprocessing and quality assessment."""

    image: np.ndarray
    page_number: int
    dpi: int
    source: str
    quality: QualityReport


@dataclass
class BatchResult:
    """Aggregated result of processing one or more documents."""

    passed: List[ProcessedPage] = field(default_factory=list)
    failed: List[ProcessedPage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed)
