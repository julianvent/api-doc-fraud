"""Aggregates rule flags into a report. Alerts only — never rejects.

Output contract:
    flags            — every fired flag
    suspicion_score  — 0.0 clean, 1.0 strongly suspicious (continuous)
    confidence       — how much metadata was available to judge
    summary          — short human-readable description
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .extractor import MetadataSnapshot
from .rules import Flag, evaluate

Confidence = Literal["high", "medium", "indeterminate"]

_SEVERITY_PENALTY: dict[str, float] = {
    "positive": 0.0,
    "low": 0.10,
    "medium": 0.25,
    "high": 0.50,
}

_HIGH_CONFIDENCE_EXIF_KEYS = 5
_OVERRIDE_CODES = {
    "TRUSTED_SIGNATURE": "Trusted government signature detected",
    "AI_SELF_DECLARATION": "File self-declares as AI-generated or synthesized",
    "ACTIVE_CONTENT": "PDF contains active content (JavaScript)",
}


@dataclass
class MetadataReport:
    source: str
    format: str
    flags: list[Flag] = field(default_factory=list)
    suspicion_score: float = 0.0
    confidence: Confidence = "indeterminate"
    summary: str = ""


def classify(snap: MetadataSnapshot) -> MetadataReport:
    flags = evaluate(snap)
    confidence = _assess_confidence(snap)
    return MetadataReport(
        source=snap.source,
        format=snap.format,
        flags=flags,
        suspicion_score=round(_compute_score(flags), 3),
        confidence=confidence,
        summary=_summarize(flags, confidence),
    )


def _assess_confidence(snap: MetadataSnapshot) -> Confidence:
    if snap.format == "pdf":
        return _pdf_confidence(snap)
    return _image_confidence(snap)


def _pdf_confidence(snap: MetadataSnapshot) -> Confidence:
    signals = sum(1 for v in (
        snap.pdf.get("producer"),
        snap.pdf.get("creator"),
        snap.pdf.get("creation_date"),
        snap.pdf.get("mod_date"),
        snap.xmp,
    ) if v)
    if signals >= 3:
        return "high"
    if signals >= 1:
        return "medium"
    return "indeterminate"


def _image_confidence(snap: MetadataSnapshot) -> Confidence:
    has_exif = bool(snap.exif)
    has_xmp = bool(snap.xmp)
    has_text = bool(snap.text_chunks)
    has_c2pa = snap.c2pa_present
    exif_keys = len(snap.exif)

    if has_c2pa or (has_exif and exif_keys >= _HIGH_CONFIDENCE_EXIF_KEYS and has_xmp):
        return "high"
    if has_exif or has_xmp or has_text:
        return "medium"
    return "indeterminate"


def _compute_score(flags: list[Flag]) -> float:
    if any(f.code == "TRUSTED_SIGNATURE" for f in flags):
        return 0.0
    penalty = sum(_SEVERITY_PENALTY.get(f.severity, 0.0) for f in flags)
    return min(penalty, 1.0)


def _summarize(flags: list[Flag], confidence: Confidence) -> str:
    for code, message in _OVERRIDE_CODES.items():
        if any(f.code == code for f in flags):
            return message

    if not flags:
        if confidence == "indeterminate":
            return "No anomalies, but insufficient metadata to assess"
        return "No anomalies, metadata internally consistent"

    severities = {f.severity for f in flags}
    codes = sorted({f.code for f in flags})
    if "high" in severities:
        return f"{len(flags)} anomaly/anomalies including high severity: {codes}"
    if "medium" in severities:
        return f"{len(flags)} notable anomaly/anomalies: {codes}"
    return f"{len(flags)} minor anomaly/anomalies: {codes}"
