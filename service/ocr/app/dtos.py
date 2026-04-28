"""Single import surface for the public DTOs of the OCR module."""
from .config import OCRConfig, VisualizationConfig, Config
from .contract import OCREngine
from .models import OCRResult, ScriptBucket, Word, Geometry

__all__ = [
    "OCRConfig",
    "VisualizationConfig",
    "Config",
    "OCREngine",
    "OCRResult",
    "ScriptBucket",
    "Word",
    "Geometry",
]
