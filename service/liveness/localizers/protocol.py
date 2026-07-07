"""FaceLocalizer Protocol — detection + crop + quality."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from service.liveness.domain.face import FaceCrop


@runtime_checkable
class FaceLocalizer(Protocol):
    """Locates the face, computes quality, and produces a crop."""

    name: str

    def warmup(self) -> None:
        ...

    def locate(self, image: np.ndarray) -> FaceCrop | None:
        """Return the best-detected face, or None if none meets min size.

        Implementations pad (default 20%) before cropping so downstream
        detectors see context, not a tight face mask.
        """
        ...
