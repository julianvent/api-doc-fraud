"""PageReport contract for the tampering module. All fields JSON-serializable.

The module produces a continuous `fraud_score`; the consumer decides policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RiskLabel(str, Enum):
    """Bucket derived from `fraud_score`: <0.30 / [0.30,0.70) / >=0.70."""
    LEGITIMATE = "LEGITIMATE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"


class Reliability(str, Enum):
    """LOW: core detector missing. MEDIUM: optional signal skipped. HIGH: all ran."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Severity(str, Enum):
    """By saturation: CRITICAL ≥1.0, HIGH ≥0.7, MEDIUM ≥0.4, LOW >0, INFO 0."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


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
    """Face localizer output. Not a forensic signal."""
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
class TruForResult:
    ran: bool
    skip_reason: Optional[str] = None
    score: float = 0.0
    largest_region_area_fraction: float = 0.0


@dataclass(frozen=True)
class DocumentLocation:
    """Axis-aligned doc crop from DocAligner. Never warped."""
    localized: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_with_padding: Optional[Tuple[int, int, int, int]] = None
    quad: Optional[Tuple[Tuple[float, float], ...]] = None
    frame_shape: Tuple[int, int] = (0, 0)
    area_fraction_of_frame: float = 0.0
    validation_status: str = "OK"
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class Artifacts:
    source_image: Optional[str] = None
    doctamper_heatmap: Optional[str] = None
    doctamper_overlay: Optional[str] = None
    trufor_heatmap: Optional[str] = None
    face_crop: Optional[str] = None
    combined_overlay: Optional[str] = None


@dataclass(frozen=True)
class Finding:
    """One reason behind `fraud_score`. `contribution` may be negative."""
    detector: str
    signal: str
    severity: Severity
    message: str
    observed_value: float
    reference_threshold: float
    contribution: float


@dataclass(frozen=True)
class Timings:
    """Wall-clock times. `per_detector` keyed by detector name."""
    total_ms: int
    per_detector: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionMetadata:
    document_type: DocumentType
    page_index: int
    page_count: int
    image_shape: Tuple[int, int]
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PageReport:
    source: str
    # Primary outputs
    fraud_score: float
    risk_label: RiskLabel
    reliability: Reliability
    findings: List[Finding]
    timings: Timings
    # Per-detector raw evidence
    face: FaceDetection
    doctamper: DocTamperResult
    trufor: TruForResult
    face_trufor: TruForResult
    # Artifacts and context
    artifacts: Artifacts
    metadata: ExecutionMetadata
    document_location: DocumentLocation = field(
        default_factory=lambda: DocumentLocation(localized=False),
    )


_LABEL_GLYPH = {
    RiskLabel.LEGITIMATE: "[LEGITIMATE]       ",
    RiskLabel.SUSPICIOUS: "[SUSPICIOUS]       ",
    RiskLabel.LIKELY_MANIPULATED: "[LIKELY_MANIPULATED]",
}


def format_report(report: PageReport) -> str:
    meta = report.metadata
    lines = [
        "=" * 72,
        f"  {report.source}  (page {meta.page_index}/{meta.page_count})",
        "=" * 72,
        f"  Fraud score : {report.fraud_score:.3f}  "
        f"{_LABEL_GLYPH[report.risk_label]} "
        f"reliability={report.reliability.value}",
        f"  Document    : {meta.document_type.value}  "
        f"shape={meta.image_shape[0]}x{meta.image_shape[1]}",
    ]

    if report.findings:
        lines.append("  Findings    :")
        for f in report.findings:
            sign = "+" if f.contribution >= 0 else ""
            lines.append(
                f"    [{f.severity.value:8s}] {f.detector}/{f.signal}  "
                f"({sign}{f.contribution:.2f})  {f.message}"
            )

    lines.append("")
    lines.append(_format_document_location(report.document_location))
    lines.append(_format_face(report.face))
    lines.append(_format_doctamper(report.doctamper))
    lines.append(_format_trufor(report.trufor, label="TruFor      "))
    lines.append(_format_trufor(report.face_trufor, label="FaceTruFor  "))

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
        artifacts.trufor_heatmap, artifacts.combined_overlay,
    ))
    if any_artifact:
        lines.append("")
        lines.append("  Artifacts:")
        if artifacts.doctamper_heatmap:
            lines.append(f"    doctamper heatmap : {artifacts.doctamper_heatmap}")
        if artifacts.doctamper_overlay:
            lines.append(f"    doctamper overlay : {artifacts.doctamper_overlay}")
        if artifacts.trufor_heatmap:
            lines.append(f"    trufor heatmap    : {artifacts.trufor_heatmap}")
        if artifacts.face_crop:
            lines.append(f"    face crop         : {artifacts.face_crop}")
        if artifacts.combined_overlay:
            lines.append(f"    combined overlay  : {artifacts.combined_overlay}")

    t = report.timings
    if t.per_detector:
        per = ", ".join(f"{k}={v}ms" for k, v in t.per_detector.items())
        lines.append(f"\n  Timings     : total={t.total_ms}ms ({per})")
    else:
        lines.append(f"\n  Timings     : total={t.total_ms}ms")

    return "\n".join(lines)


def _format_document_location(loc: DocumentLocation) -> str:
    if loc.localized and loc.bbox_with_padding is not None:
        x, y, w, h = loc.bbox_with_padding
        return (
            f"  Document    : localized  bbox=({x},{y},{w},{h})  "
            f"area_of_frame={loc.area_fraction_of_frame:.1%}"
        )
    if loc.fallback_reason:
        return f"  Document    : full-frame fallback ({loc.fallback_reason})"
    return "  Document    : full-frame (not localized)"


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


def _format_trufor(result: TruForResult, label: str = "TruFor      ") -> str:
    if not result.ran:
        return f"  {label}: skipped ({result.skip_reason})"
    return (
        f"  {label}: score={result.score:.3f}  "
        f"largest_region={result.largest_region_area_fraction:.2%}"
    )
