from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreClassResult:
    doc_family   : str
    country_iso  : Optional[str] = None
    mrz_type     : Optional[str] = None
    has_face     : bool          = False
    aspect_class : Optional[str] = None
    confidence   : float         = 0.0
    signals      : dict          = field(default_factory=dict)
