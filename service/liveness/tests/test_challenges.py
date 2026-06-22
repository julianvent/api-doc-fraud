"""Tests for the challenge taxonomy and pure criterion helpers."""

from __future__ import annotations

import pytest

from service.liveness.active import ChallengeType, DEFAULT_POOL
from service.liveness.active.challenges import CENTER_FACE_SPEC, POOL_BY_TYPE
from service.liveness.active.session import (
    _is_wrong_direction,
    _peak,
    _peak_criterion_met,
    _return_criterion_met,
)


def test_pool_composition():
    assert CENTER_FACE_SPEC.type == ChallengeType.CENTER_FACE
    assert {s.type for s in DEFAULT_POOL} == {
        ChallengeType.TURN_LEFT,
        ChallengeType.TURN_RIGHT,
        ChallengeType.BLINK,
        ChallengeType.SMILE,
    }


def test_center_face_is_single_phase():
    assert CENTER_FACE_SPEC.single_phase is True


# ---- peak criterion ----

def test_turn_left_peak_positive_yaw():
    spec = POOL_BY_TYPE[ChallengeType.TURN_LEFT]
    assert _peak_criterion_met(spec, 0.25) is True
    assert _peak_criterion_met(spec, 0.10) is False
    assert _peak_criterion_met(spec, -0.25) is False  # wrong way


def test_turn_right_peak_negative_yaw():
    spec = POOL_BY_TYPE[ChallengeType.TURN_RIGHT]
    assert _peak_criterion_met(spec, -0.25) is True
    assert _peak_criterion_met(spec, -0.10) is False
    assert _peak_criterion_met(spec, 0.25) is False


def test_blink_peak_threshold():
    spec = POOL_BY_TYPE[ChallengeType.BLINK]
    assert _peak_criterion_met(spec, 0.6) is True
    assert _peak_criterion_met(spec, 0.3) is False


def test_smile_peak_threshold():
    spec = POOL_BY_TYPE[ChallengeType.SMILE]
    assert _peak_criterion_met(spec, 0.5) is True
    assert _peak_criterion_met(spec, 0.2) is False


# ---- return criterion (round-trip) ----

@pytest.mark.parametrize("ctype", [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT])
def test_turn_return_near_zero(ctype):
    spec = POOL_BY_TYPE[ctype]
    assert _return_criterion_met(spec, 0.05) is True
    assert _return_criterion_met(spec, 0.30) is False


@pytest.mark.parametrize("ctype", [ChallengeType.BLINK, ChallengeType.SMILE])
def test_expression_return_to_neutral(ctype):
    spec = POOL_BY_TYPE[ctype]
    assert _return_criterion_met(spec, 0.05) is True
    assert _return_criterion_met(spec, 0.45) is False


# ---- peak tracking preserves the extreme value with sign ----

def test_peak_turn_right_tracks_most_negative():
    spec = POOL_BY_TYPE[ChallengeType.TURN_RIGHT]
    p = _peak(0.0, -0.20, spec)
    p = _peak(p, -0.10, spec)   # smaller magnitude, ignored
    p = _peak(p, -0.35, spec)   # larger magnitude, kept
    assert p == pytest.approx(-0.35)


def test_peak_blink_tracks_max():
    spec = POOL_BY_TYPE[ChallengeType.BLINK]
    p = _peak(0.0, 0.4, spec)
    p = _peak(p, 0.7, spec)
    p = _peak(p, 0.5, spec)
    assert p == pytest.approx(0.7)


# ---- wrong-direction only for turns ----

def test_wrong_direction_turns():
    assert _is_wrong_direction(POOL_BY_TYPE[ChallengeType.TURN_LEFT], -0.25) is True
    assert _is_wrong_direction(POOL_BY_TYPE[ChallengeType.TURN_RIGHT], 0.25) is True


def test_wrong_direction_not_for_expressions():
    assert _is_wrong_direction(POOL_BY_TYPE[ChallengeType.BLINK], -0.9) is False
    assert _is_wrong_direction(POOL_BY_TYPE[ChallengeType.SMILE], -0.9) is False
