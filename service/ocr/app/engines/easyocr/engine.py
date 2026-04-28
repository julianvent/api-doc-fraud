"""EasyOCR adapter: implements the public OCREngine contract.

Internals (detect / classify / reconstruct / agent_output) are EasyOCR-specific
helpers and stay private to this folder. The orchestrator only talks to the
contract; swapping this wrapper for another engine is purely additive.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...config import OCRConfig
from ...models import OCRResult
from . import agent_output, detect


class EasyOCREngine:
    """OCREngine implementation backed by EasyOCR's Reader."""

    name: str = "easyocr"
    version: str = "1.7"

    def __init__(self, cfg: OCRConfig) -> None:
        self._cfg = cfg
        self.languages: List[str] = list(cfg.languages)
        self._reader = None  # type: ignore[assignment]

    def warmup(self) -> None:
        if self._reader is not None:
            return
        import easyocr  # local import: heavy, only when needed

        self._reader = easyocr.Reader(
            self._cfg.languages, gpu=self._cfg.use_gpu, verbose=False,
        )

    def extract(self, image: np.ndarray, cfg: OCRConfig) -> OCRResult:
        if self._reader is None:
            self.warmup()
        words = detect.detect(image, self._reader)
        return agent_output.build(words, cfg)

    def close(self) -> None:
        self._reader = None
