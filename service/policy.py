"""Combine the 4 module outputs into a single ACCEPT/REVIEW/REJECT verdict.

Tampering contributes a continuous `fraud_score` and a `risk_label`; this
module is where weights and thresholds live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from service.metadata.analyzer import MetadataReport
from service.preprocessor.app.models import ProcessedPage
from service.tampering.detector import PageReport, RiskLabel

# Weight of each signal in the aggregate score.
_W_METADATA = 0.45
_W_TAMPERING = 0.45
_W_OCR = 0.10

_REJECT_THRESHOLD = 0.75
_REVIEW_THRESHOLD = 0.40
_METADATA_REVIEW_THRESHOLD = 0.7
_LOW_OCR_CONF = 0.5

# Rank used to find the worst tampering page.
_RISK_RANK = {
    RiskLabel.LIKELY_MANIPULATED: 2,
    RiskLabel.SUSPICIOUS: 1,
    RiskLabel.LEGITIMATE: 0,
}


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
    ocr_results: list,
) -> RiskAggregate:
    """Aggregate signals from the 4 modules into a single verdict."""
    metadata_susp = _max_metadata_suspicion(metadata_reports)
    worst_tamper_score, worst_tamper_risk = _worst_tampering(tampering_reports)
    avg_ocr_conf = _avg_ocr_confidence(ocr_results)

    score = (
        _W_METADATA * metadata_susp
        + _W_TAMPERING * worst_tamper_score
        + _W_OCR * (1.0 - avg_ocr_conf)
    )
    score = round(min(max(score, 0.0), 1.0), 3)

    verdict, reasons = _decide(score, metadata_susp, worst_tamper_risk)
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


def _worst_tampering(reports: List[PageReport]) -> tuple[float, RiskLabel]:
    if not reports:
        return 0.0, RiskLabel.LEGITIMATE
    worst = max(reports, key=lambda r: _RISK_RANK.get(r.risk_label, 0))
    worst_score = max(r.fraud_score for r in reports)
    return worst_score, worst.risk_label


def _avg_ocr_confidence(results: list) -> float:
    confidences = [
        r.get("result", {}).get("confidence_avg", 0.0)
        for r in results
        if isinstance(r, dict) and r.get("result")
    ]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


# ── decision rule ─────────────────────────────────────────────────────────────

def _decide(
    score: float, metadata_susp: float, tamper_risk: RiskLabel,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    if tamper_risk == RiskLabel.LIKELY_MANIPULATED:
        reasons.append("tampering: LIKELY_MANIPULATED on at least one page")
        return "REJECT", reasons
    if score >= _REJECT_THRESHOLD:
        reasons.append(f"aggregate score {score:.2f} >= {_REJECT_THRESHOLD}")
        return "REJECT", reasons
    if score >= _REVIEW_THRESHOLD or metadata_susp >= _METADATA_REVIEW_THRESHOLD:
        reasons.append(
            f"aggregate score {score:.2f} or metadata suspicion {metadata_susp:.2f} crosses review threshold"
        )
        return "REVIEW", reasons
    if tamper_risk == RiskLabel.SUSPICIOUS:
        reasons.append("tampering: SUSPICIOUS")
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
        if pr.risk_label != RiskLabel.LEGITIMATE:
            flags.append(f"TAMPERING_{pr.risk_label.value}")
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
