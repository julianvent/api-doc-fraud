from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


_ALLOWED_FIELD_TYPES = {
    "text", "date", "alphanumeric", "code",
    "single_letter", "entry_count", "decimal", "currency",
}


class TemplateField(BaseModel):
    key      : str
    label    : str
    type     : str            = "text"
    category : Optional[str]  = None
    # Spatial regions (normalised 0.0–1.0). Populated after the user assigns
    # elements in the dots or manual flow; absent in auto templates.
    label_region : Optional[dict] = None  # {x1, y1, x2, y2}
    value_region : Optional[dict] = None  # {x1, y1, x2, y2}
    # Element IDs sent by the client during confirm. Used to resolve the
    # regions above; stripped from the persisted template JSON.
    label_element_id : Optional[int] = None
    value_element_id : Optional[int] = None

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in _ALLOWED_FIELD_TYPES:
            raise ValueError(f"unsupported field type: {v!r}")
        return v


class TemplateSummary(BaseModel):
    id            : str
    document_type : str
    country       : Optional[str] = None
    edition       : int
    document_name : str
    created_at    : Optional[datetime] = None


class TemplateDetail(BaseModel):
    id              : str
    schema_version  : int                       = 2
    document_type   : str
    document_name   : str
    country         : Optional[str]             = None
    country_iso     : Optional[str]             = None
    state           : Optional[str]             = None
    edition         : int
    doc_family      : Optional[str]             = None
    mrz_type        : Optional[str]             = None
    img_path        : Optional[str]             = None
    fields          : list[TemplateField]       = Field(default_factory=list)
    anchors         : list[str]                 = Field(default_factory=list)
    fingerprint     : dict                      = Field(default_factory=dict)
    field_rules     : dict                      = Field(default_factory=dict)
    qr_config       : dict                      = Field(default_factory=dict)
    created_at      : Optional[datetime]        = None


# Kept for backwards compatibility with existing callers/imports.
BaseDocumentTemplateResponse = TemplateDetail