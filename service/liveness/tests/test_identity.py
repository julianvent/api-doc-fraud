"""Tests for identity-consistency verification.

Two layers: (1) pure embedding helpers (cosine, is_same_identity) with
synthetic unit vectors, no model; (2) session-level — a person swap
between challenges is caught at RECENTERING (`identity_mismatch`).
"""

from __future__ import annotations

import numpy as np

from factories import feed, synthetic_face, unit_embedding
from service.liveness.active import (
    ActiveSession,
    ChallengeType,
    SessionPhase,
    cosine_similarity,
    is_same_identity,
)
from service.liveness.active.challenges import POOL_BY_TYPE
from service.liveness.active.face_embedding import IDENTITY_MATCH_THRESHOLD


_DT = 0.05
_PREFLIGHT_FRAMES = 45


# ---- pure helpers ----

def test_cosine_self_is_one():
    e = unit_embedding(0)
    assert cosine_similarity(e, e) == pytest_approx(1.0)


def test_same_seed_is_same_identity():
    assert is_same_identity(unit_embedding(7), unit_embedding(7)) is True


def test_different_seeds_are_different_identity():
    # Random high-dim unit vectors are near-orthogonal (cosine ~0).
    assert is_same_identity(unit_embedding(0), unit_embedding(1)) is False


def test_missing_embedding_does_not_fail():
    # "Cannot evaluate" must not, by itself, reject.
    assert is_same_identity(None, unit_embedding(0)) is True
    assert is_same_identity(unit_embedding(0), None) is True
    assert is_same_identity(None, None) is True


def test_threshold_in_arcface_range():
    assert 0.25 <= IDENTITY_MATCH_THRESHOLD <= 0.40


# ---- session-level ----

def _pass_center_with_identity(s, t0, embedding):
    """Drive in-oval frames (with embedding) until CENTER_FACE passes."""
    t = t0
    for _ in range(120):
        s.submit(synthetic_face(), t, in_oval=True, embedding=embedding)
        t += _DT
        if any(r.type == ChallengeType.CENTER_FACE and r.completed for r in s.results()):
            return t
    raise AssertionError("CENTER_FACE did not pass")


def test_same_identity_completes_blink():
    person = unit_embedding(0)
    s = ActiveSession(n_challenges=2, pool=(POOL_BY_TYPE[ChallengeType.BLINK],), seed=0)
    s.start(0.0)
    t = _pass_center_with_identity(s, 0.0, person)
    # Perform the blink with the SAME identity throughout.
    feed(s, 12, t, _DT, in_oval=True, eye_blink=0.05, embedding=person); t += 12 * _DT
    feed(s, 4, t, _DT, in_oval=True, eye_blink=0.7, embedding=person); t += 4 * _DT
    snap = feed(s, 4, t, _DT, in_oval=True, eye_blink=0.05, embedding=person)
    assert snap.phase == SessionPhase.COMPLETED


def test_person_swap_caught_at_recentering():
    person_a = unit_embedding(0)
    person_b = unit_embedding(1)  # different identity
    # Two motion challenges so a RECENTERING happens between them.
    s = ActiveSession(
        n_challenges=3,
        pool=(POOL_BY_TYPE[ChallengeType.BLINK], POOL_BY_TYPE[ChallengeType.SMILE]),
        seed=0,
    )
    s.start(0.0)
    # CENTER_FACE captures person A as the identity baseline.
    t = _pass_center_with_identity(s, 0.0, person_a)
    # Complete challenge 2 as person A.
    second = s.selected_challenges[1]
    feed(s, 12, t, _DT, in_oval=True, eye_blink=0.05, mouth_smile=0.01, embedding=person_a); t += 12 * _DT
    if second == ChallengeType.BLINK:
        feed(s, 4, t, _DT, in_oval=True, eye_blink=0.7, embedding=person_a); t += 4 * _DT
        feed(s, 4, t, _DT, in_oval=True, eye_blink=0.05, embedding=person_a); t += 4 * _DT
    else:
        feed(s, 5, t, _DT, in_oval=True, mouth_smile=0.6, embedding=person_a); t += 5 * _DT
        feed(s, 5, t, _DT, in_oval=True, mouth_smile=0.05, embedding=person_a); t += 5 * _DT
    # Now in RECENTERING — swap to person B.
    snap = feed(s, 10, t, _DT, in_oval=True, embedding=person_b)
    assert snap.phase == SessionPhase.FAILED
    assert s.results()[-1].failure_reason == "identity_mismatch"


def test_no_baseline_no_identity_check():
    # No embedding ever supplied (embedder disabled): session runs
    # normally, identity check skipped.
    s = ActiveSession(n_challenges=2, pool=(POOL_BY_TYPE[ChallengeType.BLINK],), seed=0)
    s.start(0.0)
    feed(s, _PREFLIGHT_FRAMES + 12, 0.0, _DT, in_oval=True)  # no embedding
    t = (_PREFLIGHT_FRAMES + 12) * _DT
    feed(s, 12, t, _DT, in_oval=True, eye_blink=0.05); t += 12 * _DT
    feed(s, 4, t, _DT, in_oval=True, eye_blink=0.7); t += 4 * _DT
    snap = feed(s, 4, t, _DT, in_oval=True, eye_blink=0.05)
    assert snap.phase == SessionPhase.COMPLETED


# Local approx helper (avoid importing pytest at module top).
def pytest_approx(value, rel=1e-5):
    import pytest
    return pytest.approx(value, rel=rel)
