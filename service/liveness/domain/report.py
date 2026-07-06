"""Top-level report contract.

`LivenessReport` is what the facade returns; every field is JSON-
serializable via `dataclasses.asdict()`. Its shape is a versioned
contract — see `module_version` and `thresholds_version`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from service.liveness.domain.face import FaceCrop
from service.liveness.domain.verdict import Status, Verdict


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """One row of the report per detector that was attempted."""

    name: str
    status: Status
    score: float
    confidence: float
    contribution: Verdict
    reasons: tuple[str, ...]
    raw_metrics: dict[str, float]
    heatmap_path: str | None
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LivenessReport:
    """Final report returned by `liveness.analyze()`."""

    request_id: str
    timestamp: datetime
    module_version: str
    thresholds_version: str

    verdict: Verdict
    score: float
    reasons: tuple[str, ...]

    face_crop: FaceCrop | None
    detectors: dict[str, DetectorResult]
    timing_ms: dict[str, float] = field(default_factory=dict)
    evidence_dir: str | None = None
