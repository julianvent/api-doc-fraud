"""Mapping of engine name -> factory function.

Engines register themselves by being added to `_REGISTRY` in factory.py.
Adding an engine = one line here + one folder in engines/.
"""
from __future__ import annotations

from typing import Callable, Dict

from .config import OCRConfig
from .contract import OCREngine

EngineFactory = Callable[[OCRConfig], OCREngine]

_REGISTRY: Dict[str, EngineFactory] = {}


def register(name: str, factory: EngineFactory) -> None:
    """Register an engine factory under a unique name."""
    _REGISTRY[name] = factory


def get(name: str) -> EngineFactory:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown OCR engine: {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
