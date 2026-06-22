"""The PAD gate in ActiveSessionDriver: a live-person check between
centering and the motion challenges.

A fake engine controls the passive PAD verdict and per-frame signals
(centered, clean face) without loading models.
"""

from __future__ import annotations

import asyncio

import numpy as np

from factories import passive_sample, synthetic_face
from service.liveness.active import ChallengeType
from service.liveness.domain.verdict import Verdict
from service.liveness.runtime.config import RuntimeConfig
from service.liveness.runtime.frame_signals import FrameSignals
from service.liveness.runtime.session_driver import ActiveSessionDriver


class _FakeEngine:
    """Centered, clean face every frame; passive PAD returns a fixed verdict."""

    def __init__(self, passive_value=Verdict.ACCEPT):
        self._pv = passive_value

    async def process_active(self, frame, *, want_embedding=False, want_occlusion=False):
        return FrameSignals(
            face=synthetic_face(), in_oval=True,
            eye_blink=0.0, mouth_smile=0.0,
            occlusion=None, embedding=None,
        )

    async def analyze_passive(self, frame):
        return passive_sample(self._pv)


_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
_DT = 0.05


def _run(engine, frames=120):
    # Two motion challenges after CENTER_FACE for a clear boundary.
    cfg = RuntimeConfig()
    d = ActiveSessionDriver(engine, n_challenges=3, config=cfg)
    d.start(0.0)

    async def _go():
        last = None
        for i in range(frames):
            last = await d.process_frame(_FRAME, (i + 1) * _DT)
            if last.result is not None:
                break
        return last

    return d, asyncio.run(_go())


def test_spoof_rejected_at_gate_before_challenges():
    # PAD says spoof: reject right after CENTER_FACE, before any
    # motion challenge runs.
    d, last = _run(_FakeEngine(passive_value=Verdict.HARD_REJECT))
    assert last.result is not None
    assert last.result.verdict == Verdict.HARD_REJECT
    # Only CENTER_FACE should have a result; no motion challenge ran.
    completed_types = [r.type for r in d._session.results()]
    assert ChallengeType.CENTER_FACE in completed_types
    motion = {ChallengeType.BLINK, ChallengeType.SMILE,
              ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT}
    assert not any(t in motion for t in completed_types)


def test_live_person_passes_gate_and_proceeds():
    # PAD says live: gate passes and the session advances to the motion
    # challenges. It won't finish (fake face never blinks/turns) but must
    # NOT be rejected by the gate.
    d, last = _run(_FakeEngine(passive_value=Verdict.ACCEPT), frames=80)
    assert d._pad_gate_done is True
    # No early reject from the gate.
    assert d._result is None or d._result.verdict != Verdict.HARD_REJECT \
        or any(r.failure_reason for r in d._session.results())
    # The session advanced past CENTER_FACE into a motion challenge.
    assert d._session._challenge_idx >= 1
