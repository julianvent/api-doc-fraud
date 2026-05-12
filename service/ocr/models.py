from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import numpy as np

_BASE = Path(__file__).parent


@dataclass
class Config:
    confidence_threshold : float = 0.60
    ollama_url           : str   = "http://localhost:11434/api/generate"
    ollama_model         : str   = "qwen2.5:7b"
    document_fields_path : str   = str(_BASE / "data" / "document_fields.json")
    ocr_output_dir       : str   = "service/ocr/output" 

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
    birth_date  : str
    expiry_date : str
    number      : str
    sex         : str


@dataclass
class PipelineOutput:
    mrz_verified   : Optional[MRZResult]    # MRZ with valid checksum
    mrz_unverified : Optional[MRZResult]    # MRZ detected but checksum failed
    english_lines  : list[TextLine]
    english_text   : str
    source         : str
    confidence_avg : float
    raw_lines      : list[TextLine]
    template_available   : bool            = False
    template_match_score : Optional[float] = None

    @property
    def mrz(self) -> Optional[MRZResult]:
        return self.mrz_verified or self.mrz_unverified

    @property
    def has_valid_mrz(self) -> bool:
        return self.mrz_verified is not None