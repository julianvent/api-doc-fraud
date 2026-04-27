from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    confidence_threshold: float = 0.60
    max_image_side: int = 1_400
    fields_json: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "document_fields.json"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "output"
    )
