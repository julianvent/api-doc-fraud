"""Policy: combine the 4 module outputs into a single risk verdict.

This is the only place where "what to do with the signals" lives. Tweaking
weights or thresholds happens here without touching detectors or the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from service.metadata.analyzer import MetadataReport
from service.ocr.app.models import OCRResult
from service.preprocessor.app.models import ProcessedPage
from service.tampering.detector import PageReport, Verdict

# Weights of each signal in the aggregate score.
_W_METADATA = 0.45
_W_TAMPERING = 0.45
_W_OCR = 0.10

_REJECT_THRESHOLD = 0.75
_REVIEW_THRESHOLD = 0.40
_METADATA_REVIEW_THRESHOLD = 0.7
_LOW_OCR_CONF = 0.5


@dataclass
class RiskAggregate:
    score: float
    verdict: str  # ACCEPT | REVIEW | REJECT
    confidence: float
    flags: List[str]
    reasons: List[str] = field(default_factory=list)


def compute(
    metadata_reports: List[MetadataReport],
    tampering_reports: List[PageReport],
    processed_pages: List[ProcessedPage],
    ocr_results: List[OCRResult],
) -> RiskAggregate:
    """Aggregate signals from the 4 modules into a single verdict."""
    metadata_susp = _max_metadata_suspicion(metadata_reports)
    worst_tamper_score, worst_tamper_verdict = _worst_tampering(tampering_reports)
    avg_ocr_conf = _avg_ocr_confidence(ocr_results)

    score = (
        _W_METADATA * metadata_susp
        + _W_TAMPERING * worst_tamper_score
        + _W_OCR * (1.0 - avg_ocr_conf)
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    verdict, reasons = _decide(score, metadata_susp, worst_tamper_verdict)
    flags = _collect_flags(metadata_reports, tampering_reports, avg_ocr_conf)
    confidence = round(_aggregate_confidence(metadata_reports, avg_ocr_conf), 3)

    return RiskAggregate(
        score=score,
        verdict=verdict,
        confidence=confidence,
        flags=flags,
        reasons=reasons,
    )


# ── signal aggregations ───────────────────────────────────────────────────────

def _max_metadata_suspicion(reports: List[MetadataReport]) -> float:
    if not reports:
        return 0.0
    return max(r.suspicion_score for r in reports)


def _worst_tampering(reports: List[PageReport]) -> tuple[float, str]:
    if not reports:
        return 0.0, "ACCEPT"
    rank = {"HARD_REJECT": 2, "REVIEW": 1, "ACCEPT": 0}
    worst_v = max(reports, key=lambda r: rank.get(_verdict_str(r.verdict), 0))
    worst_score = max(r.verdict_score for r in reports)
    return worst_score, _verdict_str(worst_v.verdict)


def _avg_ocr_confidence(results: List[OCRResult]) -> float:
    confidences = []
    for r in results:
        for w in r.english.words:
            confidences.append(w.confidence)
        for w in r.hindi.words:
            confidences.append(w.confidence)
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _verdict_str(v) -> str:
    if isinstance(v, Verdict):
        return v.value
    return str(v)


# ── decision rule ─────────────────────────────────────────────────────────────

def _decide(score: float, metadata_susp: float, tamper_verdict: str) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if tamper_verdict == "HARD_REJECT":
        reasons.append("tampering: HARD_REJECT on at least one page")
        return "REJECT", reasons
    if score >= _REJECT_THRESHOLD:
        reasons.append(f"aggregate score {score:.2f} >= {_REJECT_THRESHOLD}")
        return "REJECT", reasons
    if score >= _REVIEW_THRESHOLD or metadata_susp >= _METADATA_REVIEW_THRESHOLD:
        reasons.append(
            f"aggregate score {score:.2f} or metadata suspicion {metadata_susp:.2f} crosses review threshold"
        )
        return "REVIEW", reasons
    if tamper_verdict == "REVIEW":
        reasons.append("tampering: REVIEW")
        return "REVIEW", reasons
    return "ACCEPT", reasons


# ── flags & confidence ────────────────────────────────────────────────────────

def _collect_flags(
    metadata_reports: List[MetadataReport],
    tampering_reports: List[PageReport],
    avg_ocr_conf: float,
) -> List[str]:
    flags: list[str] = []
    for mr in metadata_reports:
        for f in mr.flags:
            flags.append(f.code)
    for pr in tampering_reports:
        if _verdict_str(pr.verdict) != "ACCEPT":
            flags.append(f"TAMPERING_{_verdict_str(pr.verdict)}")
    if avg_ocr_conf and avg_ocr_conf < _LOW_OCR_CONF:
        flags.append("LOW_OCR_CONFIDENCE")
    return list(dict.fromkeys(flags))  # de-duplicate, keep order


def _aggregate_confidence(
    metadata_reports: List[MetadataReport],
    avg_ocr_conf: float,
) -> float:
    """A simple [0, 1] confidence score combining metadata coverage + OCR avg."""
    if not metadata_reports:
        return 0.0
    rank = {"high": 1.0, "medium": 0.6, "indeterminate": 0.3}
    md_conf = sum(rank.get(r.confidence, 0.3) for r in metadata_reports) / len(metadata_reports)
    if avg_ocr_conf == 0.0:
        return md_conf
    return 0.5 * md_conf + 0.5 * avg_ocr_conf
