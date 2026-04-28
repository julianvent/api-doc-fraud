from __future__ import annotations

from typing import List

from ...config import OCRConfig
from ...models import OCRResult, ScriptBucket, Word
from .classify import classify
from .reconstruct import reconstruct_text


def build(words: List[Word], cfg: OCRConfig) -> OCRResult:
    """Materialize the downstream agent contract from raw OCR detections.

    The shape of OCRResult IS the contract consumed by the Gemma 3 agent:
    per-script reconstructed text + word-level detail, plus a separate
    bucket for low-confidence detections. Any change here must be
    coordinated with the agent.
    """
    english, hindi, low_confidence = classify(words, cfg)
    return OCRResult(
        english=ScriptBucket(
            text=reconstruct_text(english, cfg.line_threshold),
            words=english,
        ),
        hindi=ScriptBucket(
            text=reconstruct_text(hindi, cfg.line_threshold),
            words=hindi,
        ),
        low_confidence=low_confidence,
    )
