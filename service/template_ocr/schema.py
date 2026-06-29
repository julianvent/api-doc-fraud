from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class FieldSpec(BaseModel):
    key               : str
    label             : str
    type              : str           = "text"
    category          : Optional[str] = None
    required          : bool          = False
    # Detection-first fields (Step 0). All optional so existing templates load unchanged.
    # bbox is normalized 0–1 (persisted); runtime coords live in DetectedElement.bbox.
    bbox              : Optional[list]  = None
    value_element_ids : list[int]        = Field(default_factory=list)
    label_element_id  : Optional[int]   = None


class Fingerprint(BaseModel):
    layout_desc : Optional[str] = None
    anchors     : list[str]     = Field(default_factory=list)


class QRConfig(BaseModel):
    present    : bool           = False
    signed     : bool           = False
    algorithm  : Optional[str]  = None
    format     : Optional[str]  = None
    public_key : Optional[str]  = None
    fields_map : dict           = Field(default_factory=dict)


class Template(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version    : int                    = 2
    document_type     : str                    = "unknown"
    document_name     : Optional[str]          = None
    country           : Optional[str]          = None
    country_iso       : Optional[str]          = None
    state             : Optional[str]          = None
    edition           : Optional[int]          = None
    doc_family        : Optional[str]          = None
    doc_type          : Optional[str]          = None
    issuing_authority : Optional[str]          = None
    mrz_type          : Optional[str]          = None
    year_start        : Optional[int]          = None
    year_end          : Optional[int]          = None
    img_path          : Optional[str]          = None
    anchors           : list[str]              = Field(default_factory=list)
    fingerprint       : Optional[Fingerprint]  = None
    fields            : list[FieldSpec]        = Field(default_factory=list)
    field_rules       : dict[str, dict]        = Field(default_factory=dict)
    qr_config         : Optional[QRConfig]     = None


def _flatten_v1_fields(raw_fields: Any) -> list[FieldSpec]:
    if isinstance(raw_fields, list):
        return [
            FieldSpec(
                key      = f.get("key", ""),
                label    = f.get("label", ""),
                type     = f.get("type", "text"),
                category = f.get("category"),
                required = bool(f.get("required", False)),
            )
            for f in raw_fields if isinstance(f, dict)
        ]

    if isinstance(raw_fields, dict):
        flat: list[FieldSpec] = []
        for category, group in raw_fields.items():
            if not isinstance(group, list):
                continue
            for f in group:
                if not isinstance(f, dict):
                    continue
                flat.append(FieldSpec(
                    key      = f.get("key", ""),
                    label    = f.get("label", ""),
                    type     = f.get("type", "text"),
                    category = category,
                    required = bool(f.get("required", False)),
                ))
        return flat

    return []


def load_template(raw: dict) -> Template:
    """
    Accepts a v1 (fields as {personal:[], document:[]}) or v2 (fields as flat list)
    template dict and returns a Template normalized to v2 shape.
    Normalizes country_iso to upper-case so verify-time matching is consistent.
    """
    if not isinstance(raw, dict):
        raise ValueError("template must be a dict")

    raw = {**raw}  # shallow copy so we don't mutate the caller's dict
    iso = raw.get("country_iso")
    if isinstance(iso, str):
        raw["country_iso"] = iso.strip().upper() or None

    version = int(raw.get("schema_version", 1))

    if version >= 2:
        return Template(**raw)

    return Template(
        schema_version = 1,
        document_type  = raw.get("document_type", "unknown"),
        document_name  = raw.get("document_name"),
        country        = raw.get("country"),
        country_iso    = raw.get("country_iso"),
        img_path       = raw.get("img_path"),
        fields         = _flatten_v1_fields(raw.get("fields", [])),
    )
