from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from service.ocr.paths import CONFIG_DIR, MODULE_ROOT

_DEFAULT_PATH = CONFIG_DIR / "defaults.toml"


@dataclass(frozen=True)
class OCRConfig:
    languages: List[str]
    use_gpu: bool
    confidence_threshold: float
    latin_threshold: float
    line_threshold: float


@dataclass(frozen=True)
class VisualizationConfig:
    font_path: str


@dataclass(frozen=True)
class Config:
    ocr: OCRConfig
    visualization: VisualizationConfig


def load(path: str | os.PathLike | None = None) -> Config:
    """Load OCR config from TOML. font_path is resolved against MODULE_ROOT."""
    resolved = Path(path or os.environ.get("OCR_CONFIG") or _DEFAULT_PATH)
    with resolved.open("rb") as f:
        raw = tomllib.load(f)

    raw_font = raw["visualization"]["font_path"]
    abs_font = str((MODULE_ROOT / raw_font).resolve())

    return Config(
        ocr=OCRConfig(**raw["ocr"]),
        visualization=VisualizationConfig(font_path=abs_font),
    )
