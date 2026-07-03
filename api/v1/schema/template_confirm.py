import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from api.v1.schema.document_template import TemplateField


_DOCUMENT_TYPE_RE = re.compile(r"^[a-z0-9_]{1,60}$")
_COUNTRY_ISO_RE   = re.compile(r"^[A-Z]{3}$")


class ConfirmTemplateRequest(BaseModel):
    generate_id   : Optional[str]               = None
    document_type : str
    document_name : str
    country       : Optional[str]               = None
    country_iso   : Optional[str]               = None
    state         : Optional[str]               = None
    edition       : int
    doc_family    : Optional[str]               = None
    mrz_type      : Optional[str]               = None
    fields        : list[TemplateField]
    anchors       : list[str]                   = Field(default_factory=list)
    fingerprint   : dict                        = Field(default_factory=dict)
    field_rules   : dict                        = Field(default_factory=dict)
    qr_config     : dict                        = Field(default_factory=dict)

    @field_validator("document_type")
    @classmethod
    def _check_document_type(cls, v: str) -> str:
        if not _DOCUMENT_TYPE_RE.fullmatch(v):
            raise ValueError(
                "document_type must be snake_case ASCII (a-z, 0-9, _), max 60 chars"
            )
        return v

    @field_validator("country_iso")
    @classmethod
    def _check_country_iso(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not _COUNTRY_ISO_RE.fullmatch(v):
            raise ValueError("country_iso must be 3 uppercase letters (ISO 3166-1 alpha-3)")
        return v

    @field_validator("edition")
    @classmethod
    def _check_edition(cls, v: int) -> int:
        if v < 1900 or v > 2100:
            raise ValueError("edition must be a 4-digit year between 1900 and 2100")
        return v

