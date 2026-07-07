"""Detector Protocol — semantic adapter on top of an Engine."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from service.liveness.domain.face import FaceCrop
from service.liveness.domain.report import DetectorResult


@runtime_checkable
class Detector(Protocol):
    """Runs one spoof-detection signal on a face crop.

    The detector owns its pre- and post-processing (resize, color order,
    normalization; score normalization, heatmap area, etc.).
    """

    name: str

    def warmup(self) -> None:
        """Eager-load weights so the first request does not pay the cost."""
        ...

    def analyze(self, face: FaceCrop, image_path: str) -> DetectorResult:
        """Produce a DetectorResult for the given face crop.

        `image_path` lets detectors write heatmap evidence next to the
        input; those without heatmaps ignore it.
        """
        ...
