"""Combined verdict for active liveness — binary ACCEPT / REJECT.

The final verdict is strictly binary (no REVIEW): the system runs
without a human reviewer and must commit. ACCEPT requires BOTH active
ACCEPT (every challenge completed, incl. any retry) AND passive ACCEPT
(median-of-ranks across PAD samples in ACCEPT, no sample HARD_REJECT);
anything else is REJECT, with the most actionable reason (active first,
then passive). The internal active_verdict/passive_verdict still carry
tri-valued REVIEW, kept in the report for debugging/dashboards only.
"""

from __future__ import annotations

from collections.abc import Iterable

from service.liveness.domain.active_report import ChallengeResult
from service.liveness.domain.report import LivenessReport
from service.liveness.domain.verdict import Reason, Verdict


_SEVERITY = {Verdict.ACCEPT: 0, Verdict.REVIEW: 1, Verdict.HARD_REJECT: 2}

# A passive verdict needs at least this many samples to be trustworthy.
_MIN_PASSIVE_SAMPLES = 2

# Mapping from internal failure_reason strings (set by session.py) to
# Reason enum values. Used for both motion challenges and preflight.
_FAILURE_REASON_MAP: dict[str, str] = {
    "face_lost": Reason.ACTIVE_FACE_LOST.value,
    "incomplete_motion": Reason.ACTIVE_INCOMPLETE_MOTION.value,
    "quality_failure": Reason.ACTIVE_QUALITY_FAILURE.value,
    "recentering_failed": Reason.ACTIVE_RECENTERING_FAILED.value,
    "timeout": Reason.ACTIVE_CHALLENGE_FAILED.value,
    "occluded": Reason.ACTIVE_OCCLUSION.value,
    "identity_mismatch": Reason.ACTIVE_IDENTITY_MISMATCH.value,
    "preflight_no_face": Reason.PREFLIGHT_NO_FACE.value,
    "preflight_low_quality": Reason.PREFLIGHT_LOW_QUALITY.value,
    "preflight_multiple_faces": Reason.PREFLIGHT_MULTIPLE_FACES.value,
    "preflight_occluded": Reason.PREFLIGHT_OCCLUDED.value,
}


def decide_active(
    results: Iterable[ChallengeResult],
    passive_samples: Iterable[LivenessReport],
) -> tuple[Verdict, Verdict, Verdict, tuple[str, ...], tuple[str, ...]]:
    """Return (active, passive, final, active_reasons, passive_reasons).

    `active` and `passive` are the diagnostic tri-valued verdicts (kept
    for the JSON report). `final` is the binary outcome shown to the
    user: ACCEPT or HARD_REJECT.
    """
    results = list(results)
    passive_samples = list(passive_samples)

    active_verdict, active_reasons = _active_verdict(results)
    passive_verdict, passive_reasons = _passive_verdict(passive_samples)
    final = _final_verdict(active_verdict, passive_verdict)
    return active_verdict, passive_verdict, final, active_reasons, passive_reasons


def _active_verdict(
    results: list[ChallengeResult],
) -> tuple[Verdict, tuple[str, ...]]:
    if not results:
        return Verdict.REVIEW, (Reason.ACTIVE_CHALLENGE_FAILED.value,)

    if any(r.failure_reason == "too_early" for r in results):
        return Verdict.HARD_REJECT, (Reason.ACTIVE_RESPONSE_TOO_EARLY.value,)

    if all(r.completed for r in results):
        return Verdict.ACCEPT, ()

    reasons: list[str] = []
    for r in results:
        if r.completed:
            continue
        mapped = _FAILURE_REASON_MAP.get(
            r.failure_reason or "", Reason.ACTIVE_CHALLENGE_FAILED.value
        )
        reasons.append(mapped)
    seen: set[str] = set()
    unique = tuple(r for r in reasons if not (r in seen or seen.add(r)))
    return Verdict.REVIEW, unique


def passive_verdict(
    samples: list[LivenessReport],
) -> tuple[Verdict, tuple[str, ...]]:
    """Aggregate passive PAD samples into a verdict + reasons.

    Public so the passive-only camera flow (no challenges) can reuse the
    exact same median-of-ranks aggregation the active flow uses.
    """
    return _passive_verdict(list(samples))


def _passive_verdict(
    samples: list[LivenessReport],
) -> tuple[Verdict, tuple[str, ...]]:
    if len(samples) < _MIN_PASSIVE_SAMPLES:
        # Not a spoof signal — an operational "could not verify". A
        # specific reason makes the final REJECT carry a "try again"
        # message instead of a silent reject indistinguishable from a
        # real spoof (the prior empty-reason behavior).
        return Verdict.REVIEW, (Reason.PASSIVE_INSUFFICIENT.value,)

    hard_rejects = [s for s in samples if s.verdict == Verdict.HARD_REJECT]
    if hard_rejects:
        worst = max(hard_rejects, key=lambda s: s.score)
        return Verdict.HARD_REJECT, tuple(worst.reasons)

    ranks = sorted(_SEVERITY[s.verdict] for s in samples)
    n = len(ranks)
    median = ranks[n // 2] if n % 2 else (ranks[n // 2 - 1] + ranks[n // 2]) / 2.0
    if median < 0.5:
        return Verdict.ACCEPT, ()

    review_samples = [s for s in samples if s.verdict == Verdict.REVIEW]
    if review_samples:
        worst = max(review_samples, key=lambda s: s.score)
        return Verdict.REVIEW, tuple(worst.reasons)
    return Verdict.REVIEW, ()


def _final_verdict(active: Verdict, passive: Verdict) -> Verdict:
    """Binary policy: ACCEPT only when BOTH active and passive are ACCEPT.

    No REVIEW: any uncertainty -> HARD_REJECT. The system is autonomous
    and must commit — if anything is wrong, reject.
    """
    if active == Verdict.ACCEPT and passive == Verdict.ACCEPT:
        return Verdict.ACCEPT
    return Verdict.HARD_REJECT
