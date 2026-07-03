from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np


@dataclass
class Config:
    confidence_threshold : float = 0.60
    ollama_url           : str   = "http://localhost:11434/api/generate"
    ollama_model         : str   = "qwen2.5:7b"
    ocr_output_dir       : str   = "service/ocr/output"
    templates_dir        : str   = "service/ocr/templates"

    # Change between engines
    #   OCR_ENGINE=paddle  uvicorn main:app --reload  ← default
    #   OCR_ENGINE=dots, dolphin, easy  uvicorn main:app --reload
    ocr_engine: str = field(
        default_factory=lambda: os.getenv("OCR_ENGINE", "paddle")
    )

    # VLM Models Paths
    dots_model_path: str = field(
        default_factory=lambda: os.getenv("DOTS_MODEL_PATH", "service/ocr/DotsOCR")
    )
    dolphin_model_path: str = field(
        default_factory=lambda: os.getenv("DOLPHIN_MODEL_PATH", "service/ocr/hf_model")
    )
    dolphin_repo_path: str = field(
        default_factory=lambda: os.getenv("DOLPHIN_REPO_PATH", "service/ocr/Dolphin")
    )

    # Vector matching (Qdrant + Ollama embeddings)
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "document_templates")
    )
    embedding_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    )
    match_threshold: float = field(
        default_factory=lambda: float(os.getenv("MATCH_THRESHOLD", "0.75"))
    )
    disable_vector_match: bool = field(
        default_factory=lambda: os.getenv("DISABLE_VECTOR_MATCH", "1") == "1"
    )

    # Vision backend (image → VLM for field extraction)
    ollama_vision_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_URL", "http://localhost:11434/api/generate")
    )
    ollama_vision_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "llava")
    )


@dataclass
class TextLine:
    text          : str
    confidence    : float
    bbox          : np.ndarray = field(repr=False, compare=False)
    lang          : str        = "unknown"
    original_text : Optional[str] = None
    was_mixed     : bool       = False


@dataclass
class MRZResult:
    valid       : bool
    surname     : str
    given_names : str
    country     : str
    date_of_birth  : str
    expiry_date : str
    document_number      : str
    sex         : str


@dataclass
class PipelineOutput:
    document_type  : str
    mrz_verified   : Optional[MRZResult]
    mrz_unverified : Optional[MRZResult]
    lines          : list[TextLine]
    source         : str
    confidence_avg : float

    @property
    def mrz(self) -> Optional[MRZResult]:
        return self.mrz_verified or self.mrz_unverified

    @property
    def has_valid_mrz(self) -> bool:
        return self.mrz_verified is not None