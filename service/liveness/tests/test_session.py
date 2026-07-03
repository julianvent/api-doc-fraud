"""Tests for the ActiveSession state machine.

All driven by synthetic frames — no camera, no models. Timing uses an
explicit clock (`t0 + i*dt`) so behavior is deterministic.
"""

from __future__ import annotations

import pytest

from factories import feed, synthetic_face
from service.liveness.active import ActiveSession, ChallengeType, SessionPhase
from service.liveness.active.challenges import POOL_BY_TYPE


# Frame cadence used throughout.
_DT = 0.05
# Frames to clear preflight (2 s) plus a margin.
_PREFLIGHT_FRAMES = 45


def _single(pool_type):
    """Started session with CENTER_FACE + one specific challenge."""
    s = ActiveSession(n_challenges=2, pool=(POOL_BY_TYPE[pool_type],), seed=0)
    s.start(0.0)
    return s


def _pass_center(s, t0):
    """Drive in-oval frames until CENTER_FACE passes; return the time
    after. Feeds until the CENTER_FACE result is recorded, robust to
    exact preflight/CENTER_FACE timing. Session must be started.
    """
    t = t0
    for _ in range(120):
        s.submit(synthetic_face(), t, in_oval=True, occluded=False)
        t += _DT
        if any(
            r.type == ChallengeType.CENTER_FACE and r.completed
            for r in s.results()
        ):
            return t
    raise AssertionError("CENTER_FACE did not pass within the frame budget")


# ---- construction ----

def test_center_face_always_first():
    for seed in range(6):
        s = ActiveSession(n_challenges=3, seed=seed)
        assert s.selected_challenges[0] == ChallengeType.CENTER_FACE


def test_n_challenges_validation():
    with pytest.raises(ValueError):
        ActiveSession(n_challenges=0)
    with pytest.raises(ValueError):
        ActiveSession(n_challenges=99)  # exceeds pool


def test_submit_before_start_raises():
    s = ActiveSession(n_challenges=1, seed=0)
    with pytest.raises(RuntimeError):
        s.submit(synthetic_face(), 0.0)


# ---- preflight ----

def test_preflight_pass_then_center():
    s = ActiveSession(n_challenges=1, seed=0)
    s.start(0.0)
    snap = feed(s, _PREFLIGHT_FRAMES + 12, 0.0, _DT, in_oval=True)
    assert snap.phase == SessionPhase.COMPLETED
    assert s.results()[0].type == ChallengeType.CENTER_FACE
    assert s.results()[0].completed


def test_preflight_no_face():
    s = ActiveSession(n_challenges=1, seed=0)
    s.start(0.0)
    for i in range(50):
        snap = s.submit(None, i * _DT, in_oval=False)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[0].failure_reason == "preflight_no_face"


def test_preflight_low_quality():
    s = ActiveSession(n_challenges=1, seed=0)
    s.start(0.0)
    snap = feed(s, 50, 0.0, _DT, in_oval=True, blur=0.1)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[0].failure_reason == "preflight_low_quality"


def test_preflight_multiple_faces():
    s = ActiveSession(n_challenges=1, seed=0)
    s.start(0.0)
    snap = feed(s, 50, 0.0, _DT, in_oval=True, n_faces=2)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[0].failure_reason == "preflight_multiple_faces"


def test_preflight_occluded():
    s = ActiveSession(n_challenges=1, seed=0)
    s.start(0.0)
    snap = feed(s, 50, 0.0, _DT, in_oval=True, occluded=True)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[0].failure_reason == "preflight_occluded"


# ---- round-trip motion ----

def test_turn_left_round_trip_passes():
    s = _single(ChallengeType.TURN_LEFT)
    t = _pass_center(s, 0.0)
    # Hold neutral past min_response_delay, then turn, then return.
    feed(s, 12, t, _DT, in_oval=True, nose_offset=0.0); t += 12 * _DT
    feed(s, 6, t, _DT, in_oval=True, nose_offset=0.25); t += 6 * _DT  # peak
    snap = feed(s, 6, t, _DT, in_oval=True, nose_offset=0.02)         # return
    assert snap.phase == SessionPhase.COMPLETED
    assert all(r.completed for r in s.results())


def test_peak_without_return_is_incomplete_motion():
    # Reach peak, never return. Each attempt must be CLEAN (neutral past
    # the response delay, then peak, no return) or the retry's lingering
    # peak reads as `too_early`. Two attempts -> terminal incomplete_motion.
    s = _single(ChallengeType.TURN_LEFT)
    t = _pass_center(s, 0.0)

    # Attempt 1: neutral (clears response delay) -> peak -> no return -> timeout.
    feed(s, 12, t, _DT, in_oval=True, nose_offset=0.0); t += 12 * _DT
    feed(s, 105, t, _DT, in_oval=True, nose_offset=0.25); t += 105 * _DT
    assert s._phase == SessionPhase.CHALLENGE_RETRY  # first incomplete -> retry

    # Retry hold (2 s): return to neutral so the retry doesn't read as too_early.
    feed(s, 45, t, _DT, in_oval=True, nose_offset=0.0); t += 45 * _DT

    # Attempt 2: neutral -> peak -> no return -> terminal incomplete_motion.
    feed(s, 12, t, _DT, in_oval=True, nose_offset=0.0); t += 12 * _DT
    feed(s, 105, t, _DT, in_oval=True, nose_offset=0.25)

    assert s.is_terminal
    assert s._phase == SessionPhase.FAILED
    assert s.results()[-1].failure_reason == "incomplete_motion"


def test_too_early_no_retry():
    # Hit the peak immediately (< min_response_delay) -> too_early,
    # which is NOT retryable -> terminal FAILED at once.
    s = _single(ChallengeType.TURN_LEFT)
    t = _pass_center(s, 0.0)
    snap = feed(s, 5, t, 0.03, in_oval=True, nose_offset=0.25)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[-1].failure_reason == "too_early"


# ---- retry ----

def test_timeout_grants_one_retry():
    s = _single(ChallengeType.TURN_LEFT)
    t = _pass_center(s, 0.0)
    # Never act -> window (5 s) expires -> retry phase.
    snap = feed(s, 110, t, _DT, in_oval=True, nose_offset=0.0)
    assert snap.phase == SessionPhase.CHALLENGE_RETRY


def test_retry_then_success():
    s = _single(ChallengeType.TURN_LEFT)
    t = _pass_center(s, 0.0)
    feed(s, 110, t, _DT, in_oval=True, nose_offset=0.0); t += 110 * _DT  # timeout -> retry
    feed(s, 45, t, _DT, in_oval=True, nose_offset=0.0); t += 45 * _DT    # retry hold + re-enter
    feed(s, 12, t, _DT, in_oval=True, nose_offset=0.0); t += 12 * _DT
    feed(s, 6, t, _DT, in_oval=True, nose_offset=0.25); t += 6 * _DT
    snap = feed(s, 6, t, _DT, in_oval=True, nose_offset=0.02)
    assert snap.phase == SessionPhase.COMPLETED


# ---- occlusion (#4) ----

def test_occlusion_blocks_center_face():
    s = _single(ChallengeType.BLINK)
    # Preflight clean, then occluded during CENTER_FACE for the whole
    # window + retry -> terminal occluded.
    feed(s, _PREFLIGHT_FRAMES, 0.0, _DT, in_oval=True, occluded=False)
    t = _PREFLIGHT_FRAMES * _DT
    feed(s, 400, t, _DT, in_oval=True, occluded=True)
    assert s.is_terminal
    assert s.results()[-1].failure_reason == "occluded"


def test_occlusion_during_motion_blocks():
    s = _single(ChallengeType.BLINK)
    feed(s, _PREFLIGHT_FRAMES, 0.0, _DT, in_oval=True, occluded=False)
    t = _PREFLIGHT_FRAMES * _DT
    # Pass CENTER_FACE clean.
    feed(s, 12, t, _DT, in_oval=True, occluded=False); t += 12 * _DT
    # BLINK active — occlude the whole way -> occluded.
    feed(s, 320, t, _DT, in_oval=True, occluded=True)
    assert s.is_terminal
    assert s.results()[-1].failure_reason == "occluded"


# ---- recentering between motion challenges ----

def test_recentering_runs_between_motion_challenges():
    s = ActiveSession(
        n_challenges=3,
        pool=(POOL_BY_TYPE[ChallengeType.BLINK], POOL_BY_TYPE[ChallengeType.SMILE]),
        seed=0,
    )
    s.start(0.0)
    t = _pass_center(s, 0.0)
    # Complete the first motion challenge (BLINK or SMILE).
    second = s.selected_challenges[1]
    feed(s, 12, t, _DT, in_oval=True, eye_blink=0.05, mouth_smile=0.01); t += 12 * _DT
    if second == ChallengeType.BLINK:
        feed(s, 4, t, _DT, in_oval=True, eye_blink=0.7); t += 4 * _DT
        snap = feed(s, 4, t, _DT, in_oval=True, eye_blink=0.05); t += 4 * _DT
    else:
        feed(s, 5, t, _DT, in_oval=True, mouth_smile=0.6); t += 5 * _DT
        snap = feed(s, 5, t, _DT, in_oval=True, mouth_smile=0.05); t += 5 * _DT
    assert snap.phase == SessionPhase.RECENTERING


def test_face_lost_during_challenge():
    s = _single(ChallengeType.BLINK)
    t = _pass_center(s, 0.0)
    # Submit None (no face) for > face-lost timeout (1 s).
    for i in range(40):
        snap = s.submit(None, t + i * _DT, in_oval=False)
    # face_lost is retryable -> goes to retry, then a second face_lost
    # streak fails it terminally. Drive enough frames.
    t += 40 * _DT
    for i in range(80):
        snap = s.submit(None, t + i * _DT, in_oval=False)
    assert s.is_terminal
    assert s.results()[-1].failure_reason == "face_lost"
