"""Public OCR engine contract.

Any engine plugged into this module — EasyOCR, Tesseract, a cloud API, a VLM —
must satisfy this Protocol. The orchestrator and the factory only ever talk
to OCREngine; they never import anything from `engines/<name>/` directly.

Adding a new engine:
  1. Create `engines/<name>/engine.py` exporting a class that implements OCREngine.
  2. Register it in `registry.py`.
  3. Set the engine name in config or per-request.

The output type (OCRResult) is the contract consumed by the downstream agent.
Engines must produce it; helpers in `engines/<name>/` may shape arbitrary
internals into it.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np

from .config import OCRConfig
from .models import OCRResult


class OCREngine(Protocol):
    """Minimal contract a swappable OCR engine must expose."""

    name: str
    version: str
    languages: List[str]

    def warmup(self) -> None:
        """Eagerly load models / weights. Safe to call multiple times."""
        ...

    def extract(self, image: np.ndarray, cfg: OCRConfig) -> OCRResult:
        """Run OCR on a preprocessed image and return the agent contract."""
        ...

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""
        ...
