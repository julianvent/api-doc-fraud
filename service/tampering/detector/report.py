"""Report contract for the image-tampering module.

A `PageReport` is produced for every page and contains:
  - A top-level verdict (ACCEPT / REVIEW / HARD_REJECT) with human-readable
    reasons — this is what downstream services act on.
  - Per-detector evidence (scores, regions, skip_reason) kept independent from
    the decision rule, so detectors can be swapped or added without touching
    the verdict logic.
  - Face localization output (YuNet is a localizer, not a forensic detector).
  - Paths to artifacts persisted to disk.
  - Execution metadata for observability and reproducibility.

All fields are JSON-serializable via `dataclasses.asdict()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    HARD_REJECT = "HARD_REJECT"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Zone(str, Enum):
    TEXT = "text"
    PHOTO = "photo"
    TEMPLATE = "template"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    PAN = "pan"
    INE = "ine"
    PASSPORT = "passport"
    VISA = "visa"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Region:
    """A spatial area flagged by a single detector."""
    detector: str
    bbox: Tuple[float, float, float, float]
    area_fraction: float
    score: float
    zone: Zone = Zone.UNKNOWN
    label: str = ""


@dataclass(frozen=True)
class FaceDetection:
    """Output of the face localizer. NOT a forensic signal on its own."""
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    expected: bool = True


@dataclass(frozen=True)
class DocTamperResult:
    ran: bool
    skip_reason: Optional[str] = None
    score_mean: float = 0.0
    score_max: float = 0.0
    score_outside_face: float = 0.0
    regions: List[Region] = field(default_factory=list)


@dataclass(frozen=True)
class MVSSNetResult:
    ran: bool
    skip_reason: Optional[str] = None
    # Image-level score: mean of the top-1% most suspicious pixels in the
    # face-crop heatmap. Robust to single-pixel outliers, unlike max.
    score: float = 0.0
    # Fraction of the face crop covered by the largest contiguous
    # suspicious component (pixels >= 0.5). Distinguishes one real spliced
    # region (high fraction) from scattered noise (low fraction, even when
    # the score is high).
    largest_region_area_fraction: float = 0.0


@dataclass(frozen=True)
class Artifacts:
    source_image: Optional[str] = None
    doctamper_heatmap: Optional[str] = None
    doctamper_overlay: Optional[str] = None
    mvssnet_heatmap: Optional[str] = None
    face_crop: Optional[str] = None
    combined_overlay: Optional[str] = None


@dataclass(frozen=True)
class ExecutionMetadata:
    document_type: DocumentType
    page_index: int
    page_count: int
    image_shape: Tuple[int, int]
    execution_ms: Dict[str, int] = field(default_factory=dict)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PageReport:
    source: str
    verdict: Verdict
    verdict_score: float
    verdict_reasons: List[str]
    confidence: Confidence
    face: FaceDetection
    doctamper: DocTamperResult
    mvssnet: MVSSNetResult
    artifacts: Artifacts
    metadata: ExecutionMetadata


_VERDICT_GLYPH = {
    Verdict.ACCEPT: "[ACCEPT]",
    Verdict.REVIEW: "[REVIEW]",
    Verdict.HARD_REJECT: "[REJECT]",
}


def format_report(report: PageReport) -> str:
    meta = report.metadata
    lines = [
        "=" * 72,
        f"  {report.source}  (page {meta.page_index}/{meta.page_count})",
        "=" * 72,
        f"  Verdict     : {_VERDICT_GLYPH[report.verdict]} {report.verdict.value}  "
        f"(score={report.verdict_score:.3f}, confidence={report.confidence.value})",
        f"  Document    : {meta.document_type.value}  "
        f"shape={meta.image_shape[0]}x{meta.image_shape[1]}",
    ]

    if report.verdict_reasons:
        lines.append("  Reasons     :")
        for reason in report.verdict_reasons:
            lines.append(f"    - {reason}")

    lines.append("")
    lines.append(_format_face(report.face))
    lines.append(_format_doctamper(report.doctamper))
    lines.append(_format_mvssnet(report.mvssnet))

    if report.doctamper.regions:
        lines.append("")
        lines.append(f"  Regions ({len(report.doctamper.regions)}):")
        for i, region in enumerate(report.doctamper.regions, 1):
            x1, y1, x2, y2 = region.bbox
            lines.append(
                f"    {i}. [{region.detector}/{region.zone.value}] "
                f"bbox=({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f})  "
                f"area={region.area_fraction:.2%}  score={region.score:.3f}"
            )

    artifacts = report.artifacts
    any_artifact = any((
        artifacts.doctamper_heatmap, artifacts.doctamper_overlay,
        artifacts.mvssnet_heatmap, artifacts.combined_overlay,
    ))
    if any_artifact:
        lines.append("")
        lines.append("  Artifacts:")
        if artifacts.doctamper_heatmap:
            lines.append(f"    doctamper heatmap : {artifacts.doctamper_heatmap}")
        if artifacts.doctamper_overlay:
            lines.append(f"    doctamper overlay : {artifacts.doctamper_overlay}")
        if artifacts.mvssnet_heatmap:
            lines.append(f"    mvssnet heatmap   : {artifacts.mvssnet_heatmap}")
        if artifacts.face_crop:
            lines.append(f"    face crop         : {artifacts.face_crop}")
        if artifacts.combined_overlay:
            lines.append(f"    combined overlay  : {artifacts.combined_overlay}")

    if meta.execution_ms:
        timings = ", ".join(f"{k}={v}ms" for k, v in meta.execution_ms.items())
        lines.append(f"\n  Timings     : {timings}")

    return "\n".join(lines)


def _format_face(face: FaceDetection) -> str:
    if face.detected and face.bbox is not None:
        x, y, w, h = face.bbox
        return (
            f"  Face        : detected  bbox=({x},{y},{w},{h})  "
            f"conf={face.confidence:.2f}"
        )
    if face.expected:
        return "  Face        : NOT detected (expected for this document type)"
    return "  Face        : not detected (not expected)"


def _format_doctamper(result: DocTamperResult) -> str:
    if not result.ran:
        return f"  DocTamper   : skipped ({result.skip_reason})"
    return (
        f"  DocTamper   : mean={result.score_mean:.3f}  "
        f"max={result.score_max:.3f}  "
        f"outside_face={result.score_outside_face:.3f}  "
        f"regions={len(result.regions)}"
    )


def _format_mvssnet(result: MVSSNetResult) -> str:
    if not result.ran:
        return f"  MVSS-Net    : skipped ({result.skip_reason})"
    return (
        f"  MVSS-Net    : score={result.score:.3f}  "
        f"largest_region={result.largest_region_area_fraction:.2%}"
    )
