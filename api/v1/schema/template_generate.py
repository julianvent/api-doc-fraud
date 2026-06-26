from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExpectedField(BaseModel):
    key   : str
    label : str
    type  : Optional[str] = "text"


class BBoxRegion(BaseModel):
    x1 : float
    y1 : float
    x2 : float
    y2 : float


class OCRLine(BaseModel):
    id         : int
    text       : str
    bbox       : list[list[float]]
    confidence : float


class OCRElement(BaseModel):
    id       : int
    text     : str
    category : str
    bbox     : BBoxRegion
    role     : str = "unknown"


class PreclassPayload(BaseModel):
    doc_family  : Optional[str] = None
    country_iso : Optional[str] = None
    mrz_type    : Optional[str] = None
    confidence  : Optional[float] = None


class FieldSuggestion(BaseModel):
    key            : str
    label          : str
    type           : str
    value_preview  : Optional[str]       = None
    label_line_id  : Optional[int]       = None
    value_line_ids : list[int]           = Field(default_factory=list)
    confidence     : Literal["high", "medium", "low"]
    source         : Literal["mrz", "regex", "spatial_match"]


class GenerateResponse(BaseModel):
    generate_id        : str
    expires_at         : datetime
    image_dims         : tuple[int, int]
    preclass           : PreclassPayload
    qr_config          : dict                     = Field(default_factory=dict)
    ocr_lines          : list[OCRLine]            = Field(default_factory=list)
    mrz_fields         : Optional[dict]           = None
    suggestions        : list[FieldSuggestion]    = Field(default_factory=list)
    anchors_candidates : list[str]                = Field(default_factory=list)
    # Populated only when mode='dots'; empty for auto/manual.
    ocr_elements       : list[OCRElement]         = Field(default_factory=list)
