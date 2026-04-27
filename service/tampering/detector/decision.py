"""Verdict rule: map detector evidence to ACCEPT / REVIEW / HARD_REJECT.

Decision lives in its own module on purpose. Detectors report raw evidence
(scores, regions, skip reasons); this module applies the recall-first rule
that turns evidence into a verdict. Swapping the rule — e.g. per document
type, or per tenant — does not require touching any detector.

Rule order matters: the first matching branch wins. HARD_REJECT rules run
before REVIEW rules; within a tier, strongest signal first.
"""
from __future__ import annotations

from typing import List, Tuple

from .report import (
    Confidence,
    DocTamperResult,
    FaceDetection,
    MVSSNetResult,
    Verdict,
)
from .thresholds import Thresholds


def decide(
    doctamper: DocTamperResult,
    mvssnet: MVSSNetResult,
    face: FaceDetection,
    thresholds: Thresholds,
) -> Tuple[Verdict, float, List[str], Confidence]:
    """Apply the verdict rule. Returns (verdict, score, reasons, confidence).

    `score` is the driving signal behind the verdict — the detector score that
    actually triggered the outcome — so downstream dashboards can rank pages
    by severity without having to re-derive it from the reasons list.
    """
    reasons: List[str] = []

    # ── HARD_REJECT ────────────────────────────────────────────────
    # Global DocTamper mean that high means the model sees most of the page
    # as tampered — a signature of fabricated or fully AI-generated content,
    # not natural noise. Authentic docs sit in ~0.12-0.19 even under heavy
    # compression.
    if (
        doctamper.ran
        and doctamper.score_mean >= thresholds.text_global_reject_score
    ):
        reasons.append(
            f"Global text tampering score {doctamper.score_mean:.3f} "
            f">= {thresholds.text_global_reject_score:.2f} — document appears "
            f"fabricated or fully regenerated"
        )
        return (
            Verdict.HARD_REJECT, doctamper.score_mean, reasons,
            _confidence(face, doctamper),
        )

    # Photo splicing promotes to HARD_REJECT only when BOTH the score and the
    # largest contiguous suspicious region are above threshold. High score on
    # scattered pixels is almost always scanner / security-feature noise.
    if (
        mvssnet.ran
        and mvssnet.score >= thresholds.photo_reject_score
        and mvssnet.largest_region_area_fraction >= thresholds.photo_reject_min_area_fraction
    ):
        reasons.append(
            f"Photo splicing: score={mvssnet.score:.2f} "
            f">= {thresholds.photo_reject_score:.2f} and largest suspicious "
            f"region covers {mvssnet.largest_region_area_fraction:.1%} of face "
            f"(>= {thresholds.photo_reject_min_area_fraction:.0%})"
        )
        return Verdict.HARD_REJECT, mvssnet.score, reasons, _confidence(face, doctamper)

    # Only localized strong regions count toward HARD_REJECT. Large hot
    # regions (e.g. whole header lines lit up by scanner compression) almost
    # never indicate a surgical edit — they still trigger REVIEW below.
    strong_localized_regions = [
        r for r in doctamper.regions
        if r.score >= thresholds.text_reject_score
        and r.area_fraction <= thresholds.max_region_area_fraction_for_reject
    ]
    if (
        doctamper.ran
        and len(strong_localized_regions) >= thresholds.text_reject_region_count
    ):
        reasons.append(
            f"Text tampering: {len(strong_localized_regions)} localized regions "
            f"with score >= {thresholds.text_reject_score:.2f} "
            f"(area <= {thresholds.max_region_area_fraction_for_reject:.0%})"
        )
        driving = max(r.score for r in strong_localized_regions)
        return Verdict.HARD_REJECT, driving, reasons, _confidence(face, doctamper)

    # ── REVIEW ─────────────────────────────────────────────────────
    review_score = 0.0

    # A single strong localized region is below the HARD_REJECT count but
    # still worth human review.
    if strong_localized_regions:
        peak = max(r.score for r in strong_localized_regions)
        reasons.append(
            f"{len(strong_localized_regions)} localized region(s) with "
            f"score >= {thresholds.text_reject_score:.2f}, "
            f"below HARD_REJECT count ({thresholds.text_reject_region_count})"
        )
        review_score = max(review_score, peak)

    # Strong but large regions are flagged as probable scan artifacts but not
    # auto-accepted — reviewer can confirm.
    strong_large_regions = [
        r for r in doctamper.regions
        if r.score >= thresholds.text_reject_score
        and r.area_fraction > thresholds.max_region_area_fraction_for_reject
    ]
    if strong_large_regions:
        peak = max(r.score for r in strong_large_regions)
        reasons.append(
            f"{len(strong_large_regions)} large region(s) with "
            f"score >= {thresholds.text_reject_score:.2f} "
            f"(> {thresholds.max_region_area_fraction_for_reject:.0%} area — "
            f"likely scan artifact)"
        )
        review_score = max(review_score, peak)

    if mvssnet.ran:
        # High score but the largest suspicious region is small — almost
        # certainly scanner noise. Route to REVIEW so a human confirms instead
        # of auto-accepting a genuinely ambiguous case.
        if (
            mvssnet.score >= thresholds.photo_reject_score
            and mvssnet.largest_region_area_fraction < thresholds.photo_reject_min_area_fraction
        ):
            reasons.append(
                f"Photo splicing score {mvssnet.score:.2f} but largest region "
                f"only {mvssnet.largest_region_area_fraction:.1%} of face "
                f"(below {thresholds.photo_reject_min_area_fraction:.0%}) — "
                f"likely scanner noise"
            )
            review_score = max(review_score, mvssnet.score)
        # Moderate score → review regardless of area.
        elif thresholds.photo_review_score <= mvssnet.score < thresholds.photo_reject_score:
            reasons.append(
                f"Photo splicing score {mvssnet.score:.2f} in review band "
                f"[{thresholds.photo_review_score:.2f}, {thresholds.photo_reject_score:.2f})"
            )
            review_score = max(review_score, mvssnet.score)
        # Meaningful contiguous suspicious area even with a moderate score.
        elif mvssnet.largest_region_area_fraction >= thresholds.photo_review_min_area_fraction:
            reasons.append(
                f"Largest suspicious face region covers "
                f"{mvssnet.largest_region_area_fraction:.1%} "
                f"(>= {thresholds.photo_review_min_area_fraction:.0%})"
            )
            review_score = max(review_score, mvssnet.score)

    if (
        doctamper.ran
        and doctamper.score_outside_face >= thresholds.text_review_score
    ):
        reasons.append(
            f"Text tampering outside face region: "
            f"score_outside_face={doctamper.score_outside_face:.3f} "
            f">= {thresholds.text_review_score:.2f}"
        )
        review_score = max(review_score, doctamper.score_outside_face)

    if doctamper.ran and doctamper.score_mean >= thresholds.pass_threshold:
        reasons.append(
            f"Mean text tampering score {doctamper.score_mean:.3f} "
            f">= {thresholds.pass_threshold:.2f}"
        )
        review_score = max(review_score, doctamper.score_mean)

    if face.expected and not face.detected:
        reasons.append("Face expected for this document type but not detected")
        review_score = max(review_score, 0.5)

    if reasons:
        # Ensemble promotion: a page that triggered many independent REVIEW
        # signals is collectively strong enough for HARD_REJECT, even when no
        # individual detector crossed its own reject threshold. This catches
        # fabricated documents whose evidence is spread across detectors.
        if len(reasons) >= thresholds.ensemble_reject_signal_count:
            reasons.insert(
                0,
                f"Ensemble: {len(reasons)} independent REVIEW-level signals "
                f"(>= {thresholds.ensemble_reject_signal_count})",
            )
            return (
                Verdict.HARD_REJECT, review_score, reasons,
                _confidence(face, doctamper),
            )
        return Verdict.REVIEW, review_score, reasons, _confidence(face, doctamper)

    # ── ACCEPT ──────────────────────────────────────────────────────────
    reasons.append("No detector signal above review thresholds")
    accept_score = max(
        doctamper.score_mean if doctamper.ran else 0.0,
        mvssnet.score if mvssnet.ran else 0.0,
    )
    return Verdict.ACCEPT, accept_score, reasons, _confidence(face, doctamper)


def _confidence(face: FaceDetection, doctamper: DocTamperResult) -> Confidence:
    """Heuristic confidence in the verdict itself, not in the document.

    LOW when key detectors were skipped or evidence is ambiguous; HIGH when
    every enabled detector ran and produced a clear signal.
    """
    if not doctamper.ran:
        return Confidence.LOW
    if face.expected and not face.detected:
        return Confidence.MEDIUM
    return Confidence.HIGH
