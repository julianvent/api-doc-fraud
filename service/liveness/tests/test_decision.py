"""Tests for the binary ACCEPT/REJECT decision policy."""

from __future__ import annotations

import pytest

from factories import passive_sample
from service.liveness.active import ChallengeType, decide_active
from service.liveness.active.decision import _MIN_PASSIVE_SAMPLES
from service.liveness.domain.active_report import ChallengeResult
from service.liveness.domain.verdict import Reason, Verdict


def _result(ctype, completed, reason=None):
    return ChallengeResult(
        type=ctype, completed=completed,
        detected_at_ms=500.0 if completed else None,
        peak_metric=1.0, failure_reason=reason,
    )


def _all_pass():
    return [
        _result(ChallengeType.CENTER_FACE, True),
        _result(ChallengeType.BLINK, True),
    ]


# ---- binary policy ----

def test_accept_requires_both_active_and_passive_accept():
    samples = [passive_sample(Verdict.ACCEPT)] * 4
    active, passive, final, *_ = decide_active(_all_pass(), samples)
    assert active == Verdict.ACCEPT
    assert passive == Verdict.ACCEPT
    assert final == Verdict.ACCEPT


def test_active_accept_passive_review_rejects():
    # 1 ACCEPT + 3 REVIEW -> median REVIEW -> passive REVIEW -> REJECT.
    samples = [passive_sample(Verdict.ACCEPT)] + [passive_sample(Verdict.REVIEW, 0.6)] * 3
    active, passive, final, *_ = decide_active(_all_pass(), samples)
    assert active == Verdict.ACCEPT
    assert passive == Verdict.REVIEW
    assert final == Verdict.HARD_REJECT


def test_any_passive_hard_reject_rejects():
    samples = [passive_sample(Verdict.ACCEPT)] * 5 + [passive_sample(Verdict.HARD_REJECT, 0.95)]
    _, passive, final, _, preasons = decide_active(_all_pass(), samples)
    assert passive == Verdict.HARD_REJECT
    assert final == Verdict.HARD_REJECT


def test_failed_challenge_rejects_even_with_clean_passive():
    results = [
        _result(ChallengeType.CENTER_FACE, True),
        _result(ChallengeType.BLINK, False, "timeout"),
    ]
    samples = [passive_sample(Verdict.ACCEPT)] * 4
    active, _, final, areasons, _ = decide_active(results, samples)
    assert active == Verdict.REVIEW
    assert final == Verdict.HARD_REJECT
    assert Reason.ACTIVE_CHALLENGE_FAILED.value in areasons


# ---- too_early is an anti-replay HARD_REJECT ----

def test_too_early_is_hard_reject():
    results = [_result(ChallengeType.BLINK, False, "too_early")]
    active, _, final, areasons, _ = decide_active(results, [passive_sample(Verdict.ACCEPT)] * 4)
    assert active == Verdict.HARD_REJECT
    assert final == Verdict.HARD_REJECT
    assert Reason.ACTIVE_RESPONSE_TOO_EARLY.value in areasons


# ---- insufficient passive samples (#1) ----

def test_insufficient_passive_is_explicit_reason():
    # Fewer than the minimum -> explicit PASSIVE_INSUFFICIENT, not silent.
    active, passive, final, _, preasons = decide_active(_all_pass(), [])
    assert active == Verdict.ACCEPT
    assert passive == Verdict.REVIEW
    assert final == Verdict.HARD_REJECT
    assert Reason.PASSIVE_INSUFFICIENT.value in preasons


def test_minimum_passive_samples_constant():
    # One fewer than the minimum still triggers insufficient.
    samples = [passive_sample(Verdict.ACCEPT)] * (_MIN_PASSIVE_SAMPLES - 1)
    _, _, final, _, preasons = decide_active(_all_pass(), samples)
    assert final == Verdict.HARD_REJECT
    assert Reason.PASSIVE_INSUFFICIENT.value in preasons


# ---- failure-reason mapping ----

@pytest.mark.parametrize("internal,expected", [
    ("face_lost", Reason.ACTIVE_FACE_LOST.value),
    ("incomplete_motion", Reason.ACTIVE_INCOMPLETE_MOTION.value),
    ("quality_failure", Reason.ACTIVE_QUALITY_FAILURE.value),
    ("recentering_failed", Reason.ACTIVE_RECENTERING_FAILED.value),
    ("timeout", Reason.ACTIVE_CHALLENGE_FAILED.value),
    ("occluded", Reason.ACTIVE_OCCLUSION.value),
    ("preflight_no_face", Reason.PREFLIGHT_NO_FACE.value),
    ("preflight_low_quality", Reason.PREFLIGHT_LOW_QUALITY.value),
    ("preflight_multiple_faces", Reason.PREFLIGHT_MULTIPLE_FACES.value),
    ("preflight_occluded", Reason.PREFLIGHT_OCCLUDED.value),
])
def test_failure_reason_maps_to_user_reason(internal, expected):
    results = [
        _result(ChallengeType.CENTER_FACE, True),
        _result(ChallengeType.BLINK, False, internal),
    ]
    _, _, _, areasons, _ = decide_active(results, [passive_sample(Verdict.ACCEPT)] * 4)
    assert expected in areasons


def test_empty_results_is_review_not_crash():
    active, _, final, areasons, _ = decide_active([], [passive_sample(Verdict.ACCEPT)] * 4)
    assert active == Verdict.REVIEW
    assert final == Verdict.HARD_REJECT
