from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Config:
    confidence_threshold : float = 0.60
    ollama_url           : str   = "http://localhost:11434/api/generate"
    ollama_model         : str   = "gemma3:4b"
    document_fields_path : str   = "data/document_fields.json"
    claude_api_key       : str   = ""
    openai_api_key       : str   = ""


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
    valid           : bool
    surname         : str
    given_names     : str
    country         : str
    birth_date      : str
    expiry_date     : str
    number          : str
    sex             : str


@dataclass
class PipelineOutput:
    mrz_verified   : Optional[MRZResult]    # MRZ with valid checksum
    mrz_unverified : Optional[MRZResult]    # MRZ detected but checksum failed
    english_lines  : list[TextLine]
    english_text   : str
    source         : str
    confidence_avg : float
    raw_lines      : list[TextLine]

    @property
    def mrz(self) -> Optional[MRZResult]:
        return self.mrz_verified or self.mrz_unverified

    @property
    def has_valid_mrz(self) -> bool:
        return self.mrz_verified is not None