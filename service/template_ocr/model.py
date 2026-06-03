import os
from dataclasses import dataclass, field


@dataclass
class TemplateConfig:
    ollama_url  : str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"))
    )
    ollama_timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "600"))
    )

    output_dir: str = "service/template_ocr/output"