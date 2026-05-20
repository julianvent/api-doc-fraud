from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from service.preprocessor.paths import CONFIG_DIR

_DEFAULT_PATH = CONFIG_DIR / "defaults.toml"


@dataclass(frozen=True)
class IOConfig:
    pdf_render_dpi: int


@dataclass(frozen=True)
class EnhanceConfig:
    clahe_clip_limit: float
    clahe_tile_grid: tuple[int, int]
    unsharp_sigma: float
    unsharp_amount: float
    unsharp_base: float


@dataclass(frozen=True)
class ScanConfig:
    enabled: bool
    model_filename: str
    model_input_size: int
    heatmap_threshold: float
    min_area_ratio: float
    max_area_ratio: float
    min_corner_angle_deg: float
    max_corner_angle_deg: float


@dataclass(frozen=True)
class PreprocessingConfig:
    min_dimension: int
    border_px: int
    scan: ScanConfig
    enhance: EnhanceConfig


@dataclass(frozen=True)
class QualityConfig:
    target_text_sharpness: float
    target_text_contrast: float
    target_x_height_px: float
    weight_sharpness: float
    weight_contrast: float
    weight_x_height: float
    threshold: float
    min_text_components: int
    adaptive_block_size: int
    adaptive_c: int


@dataclass(frozen=True)
class PipelineConfig:
    workers: int


@dataclass(frozen=True)
class Config:
    io: IOConfig
    preprocessing: PreprocessingConfig
    quality: QualityConfig
    pipeline: PipelineConfig


def load(path: str | os.PathLike | None = None) -> Config:
    """Load config from TOML. Precedence: argument > env PREPROCESSOR_CONFIG > default."""
    resolved = Path(path or os.environ.get("PREPROCESSOR_CONFIG") or _DEFAULT_PATH)
    with resolved.open("rb") as f:
        raw = tomllib.load(f)

    prep = raw["preprocessing"]
    scn = raw["preprocessing"]["scan"]
    enh = raw["preprocessing"]["enhance"]

    return Config(
        io=IOConfig(**raw["io"]),
        preprocessing=PreprocessingConfig(
            min_dimension=prep["min_dimension"],
            border_px=prep["border_px"],
            scan=ScanConfig(**scn),
            enhance=EnhanceConfig(
                clahe_clip_limit=enh["clahe_clip_limit"],
                clahe_tile_grid=tuple(enh["clahe_tile_grid"]),
                unsharp_sigma=enh["unsharp_sigma"],
                unsharp_amount=enh["unsharp_amount"],
                unsharp_base=enh["unsharp_base"],
            ),
        ),
        quality=QualityConfig(**raw["quality"]),
        pipeline=PipelineConfig(**raw["pipeline"]),
    )
