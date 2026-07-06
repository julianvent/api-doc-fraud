from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    confidence_threshold: float = 0.60
    max_image_side: int = 1_400
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "output"
    )
