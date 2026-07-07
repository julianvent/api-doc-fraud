"""
Document verification policy — derives ACCEPT / REVIEW / REJECT verdicts
from the combined signals of three services:

  metadata  → AI-generation / forgery artefacts in the file header
  tampering → pixel-level manipulation detection (fraud_score + risk_label)
  ocr       → layout matching, MRZ consistency, identity packet matching,
              anchor text presence, photo-region detection

Scoring rationale
─────────────────
OCR (0.40) is the primary signal because it combines the richest set of
checks: layout match confidence, MRZ integrity, identity packet comparison,
anchor text presence, and face-region detection.

Tampering (0.35) is second — pixel manipulation is a strong signal but can
yield false positives on compressed or re-scanned images.

Metadata (0.25) is supporting only — many legitimate documents are created
with tools that produce AI-looking metadata headers.

Decision flow
─────────────
Hard rules evaluate first and can short-circuit to REJECT or REVIEW based
on individual critical signals.  Only when no hard rule fires does the
aggregate numeric score determine the verdict.

  REJECT  ─ confirmed pixel manipulation, MRZ/identity data conflict,
             near-certain AI/forgery metadata, or very high aggregate score.
  REVIEW  ─ suspicious tampering, missing institutional anchors, absent
             face in photo region, invalid MRZ checksum, moderate metadata,
             or moderately elevated aggregate score.
  ACCEPT  ─ all signals within acceptable thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from service.metadata.analyzer import MetadataReport
from service.preprocessor.app.models import ProcessedPage
from service.tampering.detector import PageReport, RiskLabel


# ── weights ────────────────────────────────────────────────────────────────────
_W_OCR       = 0.40
_W_TAMPERING = 0.35
_W_METADATA  = 0.25

# ── aggregate score thresholds ─────────────────────────────────────────────────
_REJECT_THRESHOLD = 0.70
_REVIEW_THRESHOLD = 0.35

# ── per-service thresholds ─────────────────────────────────────────────────────
_METADATA_REJECT_THRESHOLD = 0.85   # near-certain AI / forgery origin
_METADATA_REVIEW_THRESHOLD = 0.60   # moderately suspicious metadata

_LOW_OCR_CONF = 0.50  # below this, OCR quality is insufficient

# ── OCR flag penalties ─────────────────────────────────────────────────────────
# Each flag adds to the OCR risk sub-score (clamped to 1.0 before weighting).
_OCR_FLAG_PENALTY: dict[str, float] = {
    "mrz_mismatch"          : 0.55,  # visible fields ≠ MRZ → very strong fraud signal
    "layout_mismatch"       : 0.35,  # document doesn't match registered template
    "suspicious_data"       : 0.30,  # VLM detected cross-field data contradiction
    "anchor_mismatch"       : 0.20,  # expected institutional text absent
    "data_inconsistency"    : 0.20,  # at least one cross-field inconsistency (VLM)
    "identity_mismatch"     : 0.20,  # identity packet values disagree with document
    "image_region_mismatch" : 0.15,  # no face where photo should be
    "mrz_checksum_failed"   : 0.10,  # MRZ integrity damaged
}

# Rank for selecting the worst tampering page when multiple pages exist.
_RISK_RANK = {
    RiskLabel.LIKELY_MANIPULATED: 2,
    RiskLabel.SUSPICIOUS:         1,
    RiskLabel.LEGITIMATE:         0,
}


@dataclass
class RiskAggregate:
    score     : float
    verdict   : str             # ACCEPT | REVIEW | REJECT
    confidence: float
    flags     : List[str]
    reasons   : List[str] = field(default_factory=list)


def compute(
    metadata_reports  : List[MetadataReport],
    tampering_reports : List[PageReport],
    processed_pages   : List[ProcessedPage],
    ocr_results       : list,
) -> RiskAggregate:
    """Aggregate signals from the three services into a single verdict."""

    metadata_susp                   = _max_metadata_suspicion(metadata_reports)
    worst_tamper_score, tamper_risk = _worst_tampering(tampering_reports)
    ocr_risk, ocr_flags             = _ocr_risk_and_flags(ocr_results)

    aggregate = round(
        min(
            _W_OCR       * ocr_risk
            + _W_TAMPERING * worst_tamper_score
            + _W_METADATA  * metadata_susp,
            1.0,
        ),
        3,
    )

    verdict, reasons = _decide(
        aggregate, metadata_susp, tamper_risk, ocr_results, ocr_flags
    )
    all_flags  = _collect_flags(metadata_reports, tampering_reports, ocr_flags, ocr_results)
    confidence = round(_aggregate_confidence(metadata_reports, ocr_results), 3)

    return RiskAggregate(
        score=aggregate,
        verdict=verdict,
        confidence=confidence,
        flags=all_flags,
        reasons=reasons,
    )


# ── OCR signal ─────────────────────────────────────────────────────────────────

def _unwrap_ocr(r: object) -> dict:
    """OCR results arrive as bare dicts (no 'result' wrapper).
    Accept both shapes for safety."""
    if not isinstance(r, dict):
        return {}
    inner = r.get("result")
    return inner if isinstance(inner, dict) else r


def _ocr_page_flags(result: dict) -> set[str]:
    """All semantic flags for a single unwrapped OCR page result.

    Extends the explicit flags list with derived flags from structured
    mismatch lists and VLM verdict so every signal surfaces in `flags`.
    """
    flags: set[str] = set(result.get("flags") or [])

    # Identity packet vs document disagreement
    if result.get("identity_mismatches"):
        flags.add("identity_mismatch")

    # Cross-field data inconsistencies detected by the VLM agent
    if result.get("inconsistencies"):
        flags.add("data_inconsistency")
        # VLM verdict "suspicious" is co-generated with inconsistencies but
        # also stands alone (e.g. future issue date detected visually).
    if result.get("verdict") == "suspicious":
        flags.add("suspicious_data")

    return flags


def ocr_page_risk(raw: object) -> float:
    """[0, 1] risk score for a single OCR page result (handles both wrapped
    and bare dict shapes). Used by the policy engine and the report builder."""
    result = _unwrap_ocr(raw)
    if not result or "error" in result:
        return 0.0

    flags = _ocr_page_flags(result)

    # Base: complement of layout-match quality (low match → high risk).
    tmc  = result.get("template_match_confidence")
    risk = (1.0 - tmc) * 0.30 if tmc is not None else 0.0

    # Additive penalty per flag
    for flag, penalty in _OCR_FLAG_PENALTY.items():
        if flag in flags:
            risk += penalty

    # Each additional identity mismatch compounds risk
    id_mm = result.get("identity_mismatches") or []
    risk += min(len(id_mm) * 0.20, 0.40)

    # Low OCR confidence degrades reliability
    ocr_conf = result.get("ocr_confidence", 1.0)
    if ocr_conf < _LOW_OCR_CONF:
        risk += 0.10

    return round(min(risk, 1.0), 3)


def _ocr_risk_and_flags(ocr_results: list) -> tuple[float, set[str]]:
    """Compute a [0, 1] OCR risk sub-score and collect all OCR flags.

    The worst page dominates (max risk across pages), mirroring the tampering
    convention where the most suspicious page drives the decision.
    """
    if not ocr_results:
        return 0.0, set()

    worst_risk: float    = 0.0
    all_flags : set[str] = set()

    for raw in ocr_results:
        result = _unwrap_ocr(raw)
        if not result or "error" in result:
            continue
        all_flags |= _ocr_page_flags(result)
        worst_risk = max(worst_risk, ocr_page_risk(raw))

    return round(worst_risk, 3), all_flags


# ── signal aggregations ────────────────────────────────────────────────────────

def _max_metadata_suspicion(reports: List[MetadataReport]) -> float:
    return max((r.suspicion_score for r in reports), default=0.0)


def _worst_tampering(reports: List[PageReport]) -> tuple[float, RiskLabel]:
    if not reports:
        return 0.0, RiskLabel.LEGITIMATE
    worst = max(reports, key=lambda r: _RISK_RANK.get(r.risk_label, 0))
    return max(r.fraud_score for r in reports), worst.risk_label


# ── decision rule ──────────────────────────────────────────────────────────────

def _decide(
    score         : float,
    metadata_susp : float,
    tamper_risk   : RiskLabel,
    ocr_results   : list,
    ocr_flags     : set[str],
) -> tuple[str, List[str]]:
    """Evaluate hard rules first, then fall back to the aggregate score.

    Hard rules short-circuit when the evidence is conclusive enough that the
    numeric score would add nothing — e.g. a proven MRZ conflict or confirmed
    pixel manipulation.
    """
    reasons: List[str] = []

    # ── REJECT — hard rules ───────────────────────────────────────────────────

    # Pixel manipulation confirmed: document integrity cannot be trusted.
    if tamper_risk == RiskLabel.LIKELY_MANIPULATED:
        reasons.append("tampering: pixel manipulation confirmed (LIKELY_MANIPULATED)")
        return "REJECT", reasons

    # Visible document data conflicts with the embedded MRZ.
    # In a genuine document both sources encode the same person; a conflict
    # means one of them has been altered.
    if "mrz_mismatch" in ocr_flags:
        reasons.append("ocr: visible field data conflicts with MRZ — possible alteration")
        return "REJECT", reasons

    # Claimant's identity packet does not match data extracted from the document.
    for raw in ocr_results:
        result = _unwrap_ocr(raw)
        id_mm  = result.get("identity_mismatches") or []
        if id_mm:
            mismatched_fields = [m.get("field") for m in id_mm]
            reasons.append(f"ocr: identity mismatch in fields {mismatched_fields}")
            return "REJECT", reasons

    # Near-certain AI-generated or forged document based on metadata.
    if metadata_susp >= _METADATA_REJECT_THRESHOLD:
        reasons.append(
            f"metadata: suspicion score {metadata_susp:.2f} >= {_METADATA_REJECT_THRESHOLD}"
            " — likely AI-generated or forged"
        )
        return "REJECT", reasons

    # Combined risk score too high.
    if score >= _REJECT_THRESHOLD:
        reasons.append(f"aggregate risk score {score:.2f} >= {_REJECT_THRESHOLD}")
        return "REJECT", reasons

    # ── REVIEW — soft rules ───────────────────────────────────────────────────

    # Pixel anomalies detected but not conclusive enough to reject.
    if tamper_risk == RiskLabel.SUSPICIOUS:
        reasons.append("tampering: suspicious pixel anomalies (SUSPICIOUS)")
        return "REVIEW", reasons

    # Document layout doesn't match any registered template — could be an
    # unknown variant, could be a substitution; human review needed.
    if "layout_mismatch" in ocr_flags:
        reasons.append("ocr: document layout does not match any registered template")
        return "REVIEW", reasons

    # Institutional anchor text (e.g. issuing authority name) absent from document.
    if "anchor_mismatch" in ocr_flags:
        reasons.append("ocr: expected institutional anchor text not found in document")
        return "REVIEW", reasons

    # No face detected where the template expects a photo.
    if "image_region_mismatch" in ocr_flags:
        reasons.append("ocr: no face detected in expected photo region")
        return "REVIEW", reasons

    # MRZ checksum invalid — MRZ may have been partially altered.
    if "mrz_checksum_failed" in ocr_flags:
        reasons.append("ocr: MRZ checksum invalid — possible partial alteration")
        return "REVIEW", reasons

    # Moderately suspicious metadata.
    if metadata_susp >= _METADATA_REVIEW_THRESHOLD:
        reasons.append(
            f"metadata: suspicion score {metadata_susp:.2f} >= {_METADATA_REVIEW_THRESHOLD}"
        )
        return "REVIEW", reasons

    # Combined risk score elevated but below reject threshold.
    if score >= _REVIEW_THRESHOLD:
        reasons.append(f"aggregate risk score {score:.2f} >= {_REVIEW_THRESHOLD}")
        return "REVIEW", reasons

    return "ACCEPT", ["all signals within acceptable thresholds"]


# ── flags & confidence ─────────────────────────────────────────────────────────

def _collect_flags(
    metadata_reports  : List[MetadataReport],
    tampering_reports : List[PageReport],
    ocr_flags         : set[str],
    ocr_results       : list,
) -> List[str]:
    flags: list[str] = []

    for mr in metadata_reports:
        for f in mr.flags:
            flags.append(f.code)

    for pr in tampering_reports:
        if pr.risk_label != RiskLabel.LEGITIMATE:
            flags.append(f"TAMPERING_{pr.risk_label.value}")

    for f in sorted(ocr_flags):
        flags.append(f"OCR_{f.upper()}")

    for raw in ocr_results:
        result = _unwrap_ocr(raw)
        ocr_conf = result.get("ocr_confidence", 1.0)
        if ocr_conf and ocr_conf < _LOW_OCR_CONF:
            flags.append("LOW_OCR_CONFIDENCE")
            break

    return list(dict.fromkeys(flags))  # deduplicate, preserve order


def _aggregate_confidence(
    metadata_reports : List[MetadataReport],
    ocr_results      : list,
) -> float:
    """[0, 1] confidence in the verdict — weighted average of signal quality.

    High confidence means the available evidence is rich and consistent.
    Low confidence (e.g. no metadata, VLM-only OCR) means the verdict
    should be treated more cautiously.
    """
    rank = {"high": 1.0, "medium": 0.6, "indeterminate": 0.3}
    md_conf = (
        sum(rank.get(r.confidence, 0.3) for r in metadata_reports) / len(metadata_reports)
        if metadata_reports else 0.5
    )

    ocr_confs = [
        c for raw in ocr_results
        if (c := _unwrap_ocr(raw).get("ocr_confidence")) is not None
    ]
    ocr_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.5

    tmc_vals = [
        v for raw in ocr_results
        if (v := _unwrap_ocr(raw).get("template_match_confidence")) is not None
    ]
    tmc_conf = sum(tmc_vals) / len(tmc_vals) if tmc_vals else 0.5

    return 0.40 * ocr_conf + 0.35 * md_conf + 0.25 * tmc_conf
