import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class TemplateTextLine:
    text     : str
    bbox     : np.ndarray = field(repr=False, compare=False)
    category : str        = "Text"


@dataclass
class TemplateConfig:
    dots_model_path: str = field(
        default_factory=lambda: os.getenv("DOTS_MODEL_PATH", "service/ocr/DotsOCR")
    )

    ollama_url  : str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma3:4b")
    )

    output_dir: str = "service/template_ocr/output"