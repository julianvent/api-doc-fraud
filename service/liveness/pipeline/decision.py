"""Decision policy — pure function, recall-first, first-match-wins.

Algorithm:

0. Hard-reject gate — no face, or face with `completeness_issues`
   (partial frame, landmark outside bbox, bad geometry); input unusable.
1. Quality gate — face below blur/brightness/size thresholds → REVIEW
   with the specific reason (usually transient capture issues).
2. Tier 1 vetoes — first detector crossing `hard_reject_score` wins
   (short-circuits the rest) → HARD_REJECT.
3. Soft signals — every detector crossing `review_score` adds its reason.
4. Ensemble promotion — if UNIQUE detectors with soft signals reach
   `hard_reject_min_unique_signals`, promote to HARD_REJECT (counts
   detectors, not len(reasons)).
5. Clean override — if all soft signals are in SOFT_REASONS and the
   `clean_override_detector` scores below `clean_override_score`,
   promote to ACCEPT. Skipped when any detector is DEGRADED.
6. Default — REVIEW if any soft signals, else ACCEPT; but a DEGRADED
   detector downgrades ACCEPT to REVIEW with DETECTOR_UNAVAILABLE (a
   human confirms rather than accepting silently with a broken detector).
"""

from __future__ import annotations

from collections.abc import Mapping

from service.liveness.config.thresholds import ThresholdsConfig
from service.liveness.domain.face import FaceCrop
from service.liveness.domain.report import DetectorResult
from service.liveness.domain.verdict import SOFT_REASONS, Reason, Status, Verdict


def decide(
    detector_results: Mapping[str, DetectorResult],
    face: FaceCrop | None,
    thresholds: ThresholdsConfig,
) -> tuple[Verdict, tuple[str, ...]]:
    """Apply the decision policy and return (verdict, reasons).

    The score is computed separately by `aggregate_score`; keeping it out
    makes the policy easier to test and swap.
    """
    has_degraded = any(
        r.status == Status.DEGRADED for r in detector_results.values()
    )

    # 0. Hard-reject gate — face missing or structurally incomplete.
    if face is None:
        return Verdict.HARD_REJECT, (Reason.FACE_NOT_DETECTED.value,)
    if face.completeness_issues:
        return Verdict.HARD_REJECT, tuple(face.completeness_issues)

    # 1. Quality gate — specific reasons matter more than the umbrella one.

    quality = face.quality
    if not quality.is_acceptable:
        reasons: list[str] = []
        q = thresholds.face_quality
        if quality.blur_score < q.blur_min:
            reasons.append(Reason.FACE_QUALITY_BLUR.value)
        if (
            quality.brightness_score < q.brightness_min
            or quality.brightness_score > q.brightness_max
        ):
            reasons.append(Reason.FACE_QUALITY_BRIGHTNESS.value)
        if face.bbox.width < q.face_min_size_px or face.bbox.height < q.face_min_size_px:
            reasons.append(Reason.FACE_TOO_SMALL.value)
        if not reasons:
            # Acceptable was False but no specific reason matched — fall back.
            reasons.append(Reason.FACE_QUALITY_LOW.value)
        return Verdict.REVIEW, tuple(reasons)

    # 2. Tier 1 vetoes — first match wins
    for name, result in detector_results.items():
        if result.status != Status.OK:
            continue
        det_thresholds = thresholds.detectors.get(name)
        if det_thresholds is None:
            continue
        if result.score >= det_thresholds.hard_reject_score:
            return Verdict.HARD_REJECT, tuple(result.reasons)

    # 3. Accumulate soft signals
    soft_reasons: list[str] = []
    soft_detectors: set[str] = set()
    for name, result in detector_results.items():
        if result.status != Status.OK:
            continue
        det_thresholds = thresholds.detectors.get(name)
        if det_thresholds is None:
            continue
        if result.score >= det_thresholds.review_score:
            soft_reasons.extend(result.reasons)
            soft_detectors.add(name)

    # 4. Ensemble promotion — count UNIQUE detectors, not len(reasons)
    if len(soft_detectors) >= thresholds.ensemble.hard_reject_min_unique_signals:
        return (
            Verdict.HARD_REJECT,
            tuple(soft_reasons) + (Reason.ENSEMBLE_PROMOTION.value,),
        )

    # 5. Clean override — disabled when degraded: the confirmer's "clean"
    # reading is less trustworthy with an incomplete ensemble.
    if soft_reasons and thresholds.ensemble.clean_override_enabled and not has_degraded:
        all_soft = all(reason in {r.value for r in SOFT_REASONS} for reason in soft_reasons)
        confirmer_name = thresholds.ensemble.clean_override_detector
        confirmer = detector_results.get(confirmer_name)
        confirmer_thresholds = thresholds.detectors.get(confirmer_name)
        if (
            all_soft
            and confirmer is not None
            and confirmer.status == Status.OK
            and confirmer_thresholds is not None
            and confirmer_thresholds.clean_override_score is not None
            and confirmer.score < confirmer_thresholds.clean_override_score
        ):
            return (
                Verdict.ACCEPT,
                tuple(soft_reasons) + (Reason.CLEAN_OVERRIDE_APPLIED.value,),
            )

    # 6. Default — downgrade ACCEPT to REVIEW when a detector is degraded.
    if soft_reasons:
        if has_degraded:
            return (
                Verdict.REVIEW,
                tuple(soft_reasons) + (Reason.DETECTOR_UNAVAILABLE.value,),
            )
        return Verdict.REVIEW, tuple(soft_reasons)
    if has_degraded:
        return Verdict.REVIEW, (Reason.DETECTOR_UNAVAILABLE.value,)
    return Verdict.ACCEPT, ()
