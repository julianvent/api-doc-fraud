"""Quantify manipulation evidence into a continuous `fraud_score` ∈ [0,1].

Does not decide ACCEPT/REVIEW/REJECT — the consumer applies its own policy.
Score is the max signal saturation (value / reject threshold), adjusted by
a counter-signal cap (clean face) and ensemble floor (many MEDIUM+ signals).
"""
from __future__ import annotations

from typing import List, Tuple

from .report import (
    DocTamperResult,
    FaceDetection,
    Finding,
    Reliability,
    RiskLabel,
    Severity,
    TruForResult,
)
from .thresholds import Thresholds


# Label cuts mirror the legacy ACCEPT / REVIEW / HARD_REJECT bands.
_LABEL_LIKELY_MANIPULATED = 0.70
_LABEL_SUSPICIOUS = 0.30
_ENSEMBLE_BOOST_SCORE = 0.75


def score(
    doctamper: DocTamperResult,
    trufor: TruForResult,
    face: FaceDetection,
    face_trufor: TruForResult,
    thresholds: Thresholds,
) -> Tuple[float, RiskLabel, Reliability, List[Finding]]:
    """Return (fraud_score, risk_label, reliability, findings)."""
    findings: List[Finding] = []
    findings.extend(_doctamper_findings(doctamper, thresholds))
    findings.extend(_trufor_findings(trufor, thresholds))
    findings.extend(_face_trufor_findings(face_trufor, thresholds))
    findings.extend(_face_findings(face))

    positive_contribs = [f.contribution for f in findings if f.contribution > 0]
    fraud_score = max(positive_contribs, default=0.0)

    # Clean face crop overrides soft signals — likely security-texture noise.
    face_confirms_clean = (
        face_trufor.ran
        and face_trufor.score < thresholds.face_trufor_clean_score
        and face_trufor.largest_region_area_fraction
        < thresholds.face_trufor_clean_area_fraction
    )
    has_hard_finding = any(
        f.severity in (Severity.HIGH, Severity.CRITICAL) for f in findings
    )
    if face_confirms_clean and not has_hard_finding and fraud_score > 0.25:
        delta = round(0.25 - fraud_score, 3)
        findings.append(Finding(
            detector="face_trufor",
            signal="counter_signal_clean_face",
            severity=Severity.INFO,
            message=(
                f"FaceTruFor confirms clean portrait (score={face_trufor.score:.3f}, "
                f"area={face_trufor.largest_region_area_fraction:.1%}) — soft signals "
                f"likely security-texture noise"
            ),
            observed_value=face_trufor.score,
            reference_threshold=thresholds.face_trufor_clean_score,
            contribution=delta,
        ))
        fraud_score = 0.25

    # Ensemble boost: many co-occurring MEDIUM+ signals lift the score.
    medium_plus = [
        f for f in findings
        if f.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
    ]
    if (
        len(medium_plus) >= thresholds.ensemble_reject_signal_count
        and fraud_score < _ENSEMBLE_BOOST_SCORE
    ):
        delta = round(_ENSEMBLE_BOOST_SCORE - fraud_score, 3)
        findings.insert(0, Finding(
            detector="ensemble",
            signal="ensemble_promotion",
            severity=Severity.HIGH,
            message=(
                f"{len(medium_plus)} independent MEDIUM+ findings "
                f"(>= {thresholds.ensemble_reject_signal_count}) — evidence is "
                f"collectively strong"
            ),
            observed_value=float(len(medium_plus)),
            reference_threshold=float(thresholds.ensemble_reject_signal_count),
            contribution=delta,
        ))
        fraud_score = _ENSEMBLE_BOOST_SCORE

    fraud_score = max(0.0, min(1.0, fraud_score))
    return (
        round(fraud_score, 3),
        _label_from_score(fraud_score),
        _reliability(doctamper, trufor, face_trufor, face),
        findings,
    )


def _doctamper_findings(
    dt: DocTamperResult, t: Thresholds,
) -> List[Finding]:
    if not dt.ran:
        return [Finding(
            detector="doctamper",
            signal="skipped",
            severity=Severity.INFO,
            message=f"DocTamper did not run ({dt.skip_reason})",
            observed_value=0.0,
            reference_threshold=0.0,
            contribution=0.0,
        )]

    findings: List[Finding] = []

    # DT mean alone does not discriminate (Real and Fake both ~0.32);
    # only contribute when at or above the review entry.
    if dt.score_mean >= t.text_global_reject_score:
        findings.append(Finding(
            detector="doctamper",
            signal="score_mean_reject",
            severity=Severity.CRITICAL,
            message=(
                f"DocTamper mean score {dt.score_mean:.3f} "
                f">= {t.text_global_reject_score:.2f} (Tier-1 reject) — "
                f"document appears fabricated or fully regenerated"
            ),
            observed_value=dt.score_mean,
            reference_threshold=t.text_global_reject_score,
            contribution=1.0,
        ))
    elif dt.score_mean >= t.pass_threshold:
        contribution = min(0.65, dt.score_mean / t.text_global_reject_score)
        findings.append(Finding(
            detector="doctamper",
            signal="score_mean_elevated",
            severity=_severity_from_saturation(contribution),
            message=(
                f"DocTamper mean score {dt.score_mean:.3f} "
                f">= {t.pass_threshold:.2f} (review entry)"
            ),
            observed_value=dt.score_mean,
            reference_threshold=t.pass_threshold,
            contribution=round(contribution, 3),
        ))

    # Outside-face score: review-only, capped below the reject cut.
    if dt.score_outside_face >= t.text_review_score:
        contribution = min(0.60, dt.score_outside_face / t.text_global_reject_score)
        findings.append(Finding(
            detector="doctamper",
            signal="score_outside_face",
            severity=Severity.MEDIUM,
            message=(
                f"Text tampering outside face region: "
                f"{dt.score_outside_face:.3f} >= {t.text_review_score:.2f}"
            ),
            observed_value=dt.score_outside_face,
            reference_threshold=t.text_review_score,
            contribution=round(contribution, 3),
        ))

    # Strong localized regions: Tier-1 reject when ≥ N together.
    strong_localized = [
        r for r in dt.regions
        if r.score >= t.text_reject_score
        and r.area_fraction <= t.max_region_area_fraction_for_reject
    ]
    if strong_localized:
        peak = max(r.score for r in strong_localized)
        if len(strong_localized) >= t.text_reject_region_count:
            contribution = 1.0
            severity = Severity.CRITICAL
            msg = (
                f"{len(strong_localized)} localized regions with score >= "
                f"{t.text_reject_score:.2f} (Tier-1 reject)"
            )
        else:
            contribution = 0.65
            severity = Severity.MEDIUM
            msg = (
                f"{len(strong_localized)} localized region(s) with score >= "
                f"{t.text_reject_score:.2f} (below count threshold "
                f"{t.text_reject_region_count})"
            )
        findings.append(Finding(
            detector="doctamper",
            signal="strong_localized_regions",
            severity=severity,
            message=msg,
            observed_value=peak,
            reference_threshold=t.text_reject_score,
            contribution=contribution,
        ))

    # Large strong regions: usually scan artifact, not a surgical edit.
    strong_large = [
        r for r in dt.regions
        if r.score >= t.text_reject_score
        and r.area_fraction > t.max_region_area_fraction_for_reject
    ]
    if strong_large:
        peak = max(r.score for r in strong_large)
        findings.append(Finding(
            detector="doctamper",
            signal="strong_large_regions",
            severity=Severity.MEDIUM,
            message=(
                f"{len(strong_large)} large region(s) with score >= "
                f"{t.text_reject_score:.2f} (likely scan artifact)"
            ),
            observed_value=peak,
            reference_threshold=t.text_reject_score,
            contribution=0.55,
        ))

    return findings


def _trufor_findings(tf: TruForResult, t: Thresholds) -> List[Finding]:
    if not tf.ran:
        return [Finding(
            detector="trufor",
            signal="skipped",
            severity=Severity.INFO,
            message=f"TruFor did not run ({tf.skip_reason})",
            observed_value=0.0,
            reference_threshold=0.0,
            contribution=0.0,
        )]

    findings: List[Finding] = []

    # Tier 1: high score AND meaningful area.
    if (
        tf.score >= t.trufor_reject_score
        and tf.largest_region_area_fraction >= t.trufor_reject_min_area_fraction
    ):
        findings.append(Finding(
            detector="trufor",
            signal="high_score_high_area",
            severity=Severity.CRITICAL,
            message=(
                f"TruFor score {tf.score:.2f} >= {t.trufor_reject_score:.2f} and "
                f"suspicious region covers {tf.largest_region_area_fraction:.1%} "
                f"(>= {t.trufor_reject_min_area_fraction:.0%})"
            ),
            observed_value=tf.score,
            reference_threshold=t.trufor_reject_score,
            contribution=1.0,
        ))
        return findings

    # High score, small area: likely scanner noise — capped below reject.
    if (
        tf.score >= t.trufor_reject_score
        and tf.largest_region_area_fraction < t.trufor_reject_min_area_fraction
    ):
        contribution = min(0.65, tf.score / t.trufor_reject_score * 0.65)
        findings.append(Finding(
            detector="trufor",
            signal="high_score_low_area",
            severity=Severity.HIGH,
            message=(
                f"TruFor score {tf.score:.2f} but largest region only "
                f"{tf.largest_region_area_fraction:.1%} of doc (likely scanner noise)"
            ),
            observed_value=tf.score,
            reference_threshold=t.trufor_reject_score,
            contribution=round(contribution, 3),
        ))
        return findings

    # Score in review band: capped below the reject cut.
    if t.trufor_review_score <= tf.score < t.trufor_reject_score:
        contribution = min(0.65, tf.score / t.trufor_reject_score * 0.78)
        findings.append(Finding(
            detector="trufor",
            signal="score_review_band",
            severity=Severity.HIGH,
            message=(
                f"TruFor score {tf.score:.2f} in review band "
                f"[{t.trufor_review_score:.2f}, {t.trufor_reject_score:.2f})"
            ),
            observed_value=tf.score,
            reference_threshold=t.trufor_review_score,
            contribution=round(contribution, 3),
        ))
        return findings

    # Area-only signal: Real/Fake distributions overlap heavily, so the
    # contribution is dampened (floor 0.25, ceiling 0.60).
    if tf.largest_region_area_fraction >= t.trufor_review_min_area_fraction:
        area_ratio = (
            (tf.largest_region_area_fraction - t.trufor_review_min_area_fraction)
            / max(
                t.trufor_reject_min_area_fraction - t.trufor_review_min_area_fraction,
                1e-6,
            )
        )
        contribution = min(0.60, 0.25 + max(0.0, min(1.0, area_ratio)) * 0.35)
        findings.append(Finding(
            detector="trufor",
            signal="area_only",
            severity=Severity.MEDIUM,
            message=(
                f"TruFor suspicious area {tf.largest_region_area_fraction:.1%} "
                f"(>= {t.trufor_review_min_area_fraction:.0%}) with moderate score "
                f"{tf.score:.2f}"
            ),
            observed_value=tf.largest_region_area_fraction,
            reference_threshold=t.trufor_review_min_area_fraction,
            contribution=round(contribution, 3),
        ))

    return findings


def _face_trufor_findings(
    ftf: TruForResult, t: Thresholds,
) -> List[Finding]:
    if not ftf.ran:
        return [Finding(
            detector="face_trufor",
            signal="skipped",
            severity=Severity.INFO,
            message=f"FaceTruFor did not run ({ftf.skip_reason})",
            observed_value=0.0,
            reference_threshold=0.0,
            contribution=0.0,
        )]

    # Tier 1: face manipulation (score AND area).
    if (
        ftf.score >= t.face_trufor_reject_score
        and ftf.largest_region_area_fraction >= t.face_trufor_reject_min_area_fraction
    ):
        return [Finding(
            detector="face_trufor",
            signal="face_manipulation",
            severity=Severity.CRITICAL,
            message=(
                f"TruFor on face crop score={ftf.score:.2f} >= "
                f"{t.face_trufor_reject_score:.2f} and suspicious area covers "
                f"{ftf.largest_region_area_fraction:.1%} (>= "
                f"{t.face_trufor_reject_min_area_fraction:.0%})"
            ),
            observed_value=ftf.score,
            reference_threshold=t.face_trufor_reject_score,
            contribution=1.0,
        )]

    # Below Tier 1: emit as INFO with zero contribution; the clean-face
    # counter-signal in `score()` handles the negative case.
    return [Finding(
        detector="face_trufor",
        signal="face_below_threshold",
        severity=Severity.INFO,
        message=(
            f"TruFor on face crop score={ftf.score:.3f}, "
            f"area={ftf.largest_region_area_fraction:.1%} "
            f"(below reject {t.face_trufor_reject_score:.2f} / "
            f"{t.face_trufor_reject_min_area_fraction:.0%})"
        ),
        observed_value=ftf.score,
        reference_threshold=t.face_trufor_reject_score,
        contribution=0.0,
    )]


def _face_findings(face: FaceDetection) -> List[Finding]:
    if face.expected and not face.detected:
        return [Finding(
            detector="face_localizer",
            signal="face_expected_missing",
            severity=Severity.MEDIUM,
            message="Face expected for this document type but not detected",
            observed_value=0.0,
            reference_threshold=1.0,
            contribution=0.5,
        )]
    return []


def _label_from_score(score: float) -> RiskLabel:
    if score >= _LABEL_LIKELY_MANIPULATED:
        return RiskLabel.LIKELY_MANIPULATED
    if score >= _LABEL_SUSPICIOUS:
        return RiskLabel.SUSPICIOUS
    return RiskLabel.LEGITIMATE


def _severity_from_saturation(saturation: float) -> Severity:
    if saturation >= 1.0:
        return Severity.CRITICAL
    if saturation >= 0.7:
        return Severity.HIGH
    if saturation >= 0.4:
        return Severity.MEDIUM
    if saturation > 0:
        return Severity.LOW
    return Severity.INFO


def _reliability(
    doctamper: DocTamperResult,
    trufor: TruForResult,
    face_trufor: TruForResult,
    face: FaceDetection,
) -> Reliability:
    """LOW when a core detector skipped; MEDIUM when an optional one did."""
    if not doctamper.ran or not trufor.ran:
        return Reliability.LOW
    if not face_trufor.ran:
        return Reliability.MEDIUM
    if face.expected and not face.detected:
        return Reliability.MEDIUM
    return Reliability.HIGH
