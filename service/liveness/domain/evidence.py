"""Raw evidence produced by inference engines.

EngineOutput is what an `Engine.infer()` returns. Adapters in
`detectors/` translate it into a `DetectorResult`. Heatmaps stay as
file paths in the report — pixel data lives on disk, not in JSON.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeatmapData:
    """Spatial evidence from some engines (e.g. DeepPixBiS).

    `area_fraction` is the fraction of pixels above the engine's internal
    threshold — kept here so the adapter applies rules without re-loading
    the heatmap from disk.
    """

    heatmap_path: str
    area_fraction: float
    mean_intensity: float
    max_intensity: float


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """Result of a single inference call."""

    score: float
    confidence: float
    raw_metrics: dict[str, float]
    heatmap: HeatmapData | None = None
