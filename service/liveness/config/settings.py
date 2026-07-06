"""Runtime settings for the liveness module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Behavior knobs that vary by environment / deployment."""

    enabled_detectors: tuple[str, ...]
    face_crop_padding: float
    face_min_size_px: int
    use_onnx: bool
    persist_artifacts: bool


DEFAULT_SETTINGS = Settings(
    enabled_detectors=("moire", "minifas"),
    face_crop_padding=0.20,
    face_min_size_px=64,
    use_onnx=False,
    persist_artifacts=True,
)
