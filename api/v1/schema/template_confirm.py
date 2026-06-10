from typing import Optional

from pydantic import BaseModel, Field, model_validator

from api.v1.schema.document_template import TemplateField


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

    @model_validator(mode="after")
    def _unique_keys(self):
        seen: set[str] = set()
        for f in self.fields:
            if f.key in seen:
                raise ValueError(f"duplicate field key: {f.key!r}")
            seen.add(f.key)
        return self
