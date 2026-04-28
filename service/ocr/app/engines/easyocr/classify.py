from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from ...config import OCRConfig
from ...models import Word

# Devanagari block starts at U+0900. Everything below is Latin / digits / punct.
_DEVANAGARI_START = 0x0900


def classify(
    words: List[Word], cfg: OCRConfig,
) -> Tuple[List[Word], List[Word], List[Word]]:
    """Split words into english / hindi / low_confidence buckets.

    Mixed-script words are split character-by-character and the resulting
    fragments are placed in both the english and hindi buckets, preserving
    the original geometry so the caller can still group them spatially.
    """
    english: List[Word] = []
    hindi: List[Word] = []
    low_conf: List[Word] = []

    for w in words:
        if w.confidence < cfg.confidence_threshold:
            low_conf.append(w)
            continue

        script = classify_script(w.text, cfg.latin_threshold)
        if script == "english":
            english.append(w)
        elif script == "hindi":
            hindi.append(w)
        else:
            eng_text, hin_text = split_mixed_word(w.text)
            if eng_text:
                english.append(replace(w, text=eng_text, original=w.text))
            if hin_text:
                hindi.append(replace(w, text=hin_text, original=w.text))

    return english, hindi, low_conf


def classify_script(text: str, latin_threshold: float) -> str:
    """Return 'english', 'hindi', or 'mixed' based on character Unicode ranges."""
    if not text.strip():
        return "english"
    latin_chars = sum(1 for c in text if ord(c) < _DEVANAGARI_START)
    fraction = latin_chars / len(text)
    if fraction >= latin_threshold:
        return "english"
    if fraction == 0.0:
        return "hindi"
    return "mixed"


def split_mixed_word(text: str) -> Tuple[str, str]:
    """Split a mixed-script token into its Latin and Devanagari portions."""
    english_chars = [c for c in text if ord(c) < _DEVANAGARI_START]
    hindi_chars = [c for c in text if ord(c) >= _DEVANAGARI_START]
    return "".join(english_chars).strip(), "".join(hindi_chars).strip()
