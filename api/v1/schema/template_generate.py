from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExpectedField(BaseModel):
    key   : str
    label : str
    type  : Optional[str] = "text"


class OCRLine(BaseModel):
    id         : int
    text       : str
    bbox       : list[list[float]]
    confidence : float


class PreclassPayload(BaseModel):
    doc_family  : Optional[str] = None
    country_iso : Optional[str] = None
    mrz_type    : Optional[str] = None
    confidence  : Optional[float] = None


class FieldSuggestion(BaseModel):
    key               : str
    label             : str
    type              : str
    value_preview     : Optional[str]  = None
    label_element_id  : Optional[int]  = None
    value_element_ids : list[int]      = Field(default_factory=list)
    confidence        : Literal["high", "medium", "low"]
    source            : Literal["mrz", "regex", "spatial_match"]


class BBoxRegion(BaseModel):
    """Normalised bounding box (0.0–1.0 relative to image dimensions)."""
    x1 : float
    y1 : float
    x2 : float
    y2 : float


class OCRElement(BaseModel):
    """A single text element returned by DotsOCR in mode='dots'.

    The id is stable within a generate/confirm session: pass it back as
    label_element_id or value_element_id in the confirm request so the
    API can resolve the spatial region for that field.

    role is a best-effort hint: 'label' for field names, 'value' for field
    values, 'unknown' when there is not enough signal. The user can override
    this in the UI before confirming.
    """
    id       : int
    text     : str
    bbox     : BBoxRegion
    category : str
    role     : Literal["label", "value", "unknown"] = "unknown"


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
