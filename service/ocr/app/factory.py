"""Build an OCREngine by name. Registers built-in engines on import."""
from __future__ import annotations

from .config import OCRConfig
from .contract import OCREngine
from .registry import get, register
from .engines.easyocr.engine import EasyOCREngine

# Built-in engines. New engines add a single line here.
register("easyocr", lambda cfg: EasyOCREngine(cfg))


def build_engine(name: str, cfg: OCRConfig) -> OCREngine:
    """Instantiate the engine identified by `name` with the given config."""
    factory = get(name)
    return factory(cfg)
