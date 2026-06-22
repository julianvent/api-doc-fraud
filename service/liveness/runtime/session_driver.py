"""ActiveSessionDriver — transport-agnostic orchestration of one session.

The webcam demo's SESSION per-frame logic, ported to async and decoupled
from camera/render. Any transport feeds it frames and forwards the
updates it returns. Responsibilities: throttle + cache the expensive
parser (occlusion); compute the identity embedding only in neutral-pose
phases; sample passive PAD in stable windows; enforce a whole-session
timeout; finalize via `decide_active`, picking the user message
(occlusion-specific when applicable). It owns the per-session mutable
state; the shared `InferenceEngine` (models) is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from service.liveness.active import (
    ActiveSession,
    ChallengeType,
    SessionPhase,
    decide_active,
    passive_verdict,
    primary_message_for,
)
from service.liveness.active.occlusion import FaceOcclusionReport
from service.liveness.domain.verdict import Reason, Verdict
from service.liveness.runtime.config import DEFAULT_RUNTIME_CONFIG, RuntimeConfig
from service.liveness.runtime.inference_engine import InferenceEngine


_OCCLUSION_REASONS = {
    Reason.ACTIVE_OCCLUSION.value,
    Reason.PREFLIGHT_OCCLUDED.value,
}
_ACCEPT_MESSAGE = "Liveness verified successfully."
_TIMEOUT_MESSAGE = "The check took too long. Please try again."
_NOT_LIVE_MESSAGE = "Not a live person detected. Use your real face, not a photo or screen."


@dataclass(frozen=True, slots=True)
class DriverResult:
    """Terminal outcome of a session."""

    verdict: Verdict        # module verdict: ACCEPT or HARD_REJECT
    reasons: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class DriverUpdate:
    """What the transport sends to the client after one frame."""

    snapshot: object                 # SessionSnapshot
    in_oval: bool                    # face centered+sized in the guide oval
    occlusion_hint: str | None
    result: DriverResult | None      # set once the session is terminal


class ActiveSessionDriver:
    def __init__(
        self,
        engine: InferenceEngine,
        *,
        n_challenges: int,
        config: RuntimeConfig = DEFAULT_RUNTIME_CONFIG,
        seed: int | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._session = ActiveSession(n_challenges=n_challenges, seed=seed)
        self._started_at = datetime.now(timezone.utc)
        self._started_monotonic: float | None = None

        # Per-session orchestration state (mirrors the demo).
        self._last_parser_at = float("-inf")
        self._embedding_relevant = True
        self._last_occlusion: FaceOcclusionReport | None = None
        self._last_occlusion_hint: str | None = None
        self._last_passive_at = float("-inf")
        self._passive_samples: list = []
        self._last_completed = 0
        self._pad_gate_done = False
        self._result: DriverResult | None = None

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def selected_challenges(self):
        return self._session.selected_challenges

    def start(self, now_s: float) -> None:
        self._started_monotonic = now_s
        self._session.start(now_s)

    async def process_frame(self, frame_bgr: np.ndarray, now_s: float) -> DriverUpdate:
        # Already finished — return the cached result.
        if self._result is not None:
            return DriverUpdate(
                self._session.submit(None, now_s), False,
                self._last_occlusion_hint, self._result,
            )

        # Global session timeout (covers retries that could run past
        # the budget).
        if (
            self._started_monotonic is not None
            and now_s - self._started_monotonic > self._config.session_timeout_s
        ):
            self._result = DriverResult(
                Verdict.HARD_REJECT,
                (Reason.ACTIVE_CHALLENGE_FAILED.value,),
                _TIMEOUT_MESSAGE,
            )
            return DriverUpdate(self._session.submit(None, now_s), False, None, self._result)

        # Decide which expensive signals to compute this frame.
        want_occlusion = (
            (now_s - self._last_parser_at) * 1000.0 >= self._config.parser_throttle_ms
        )
        want_embedding = self._embedding_relevant

        signals = await self._engine.process_active(
            frame_bgr, want_embedding=want_embedding, want_occlusion=want_occlusion
        )

        # Refresh the throttled occlusion cache.
        if want_occlusion and signals.occlusion is not None:
            self._last_occlusion = signals.occlusion
            self._last_parser_at = now_s
        occlusion = self._last_occlusion
        occluded = occlusion is not None and occlusion.has_occlusion
        if occluded:
            self._last_occlusion_hint = occlusion.hint()

        snap = self._session.submit(
            signals.face,
            now_s,
            in_oval=signals.in_oval,
            eye_blink=signals.eye_blink,
            mouth_smile=signals.mouth_smile,
            occluded=occluded,
            embedding=signals.embedding,
        )

        await self._maybe_sample_passive(snap, frame_bgr, now_s)

        # Embedding matters next frame only in neutral-pose phases.
        self._embedding_relevant = snap.phase in (
            SessionPhase.PREFLIGHT,
            SessionPhase.RECENTERING,
        ) or (
            snap.phase == SessionPhase.CHALLENGE_ACTIVE
            and snap.current_challenge is not None
            and snap.current_challenge.type == ChallengeType.CENTER_FACE
        )

        # PAD GATE: once CENTER_FACE passes, confirm a LIVE person before
        # the motion challenges — a photo/replay is rejected here, never
        # asked to blink or turn. The ~6+ passive samples from the 2 s
        # preflight give enough evidence to judge.
        gate = self._maybe_pad_gate(now_s)
        if gate is not None:
            return DriverUpdate(snap, signals.in_oval, gate, self._result)

        if self._session.is_terminal:
            self._result = self._finalize()

        hint = occlusion.hint() if occluded else None
        return DriverUpdate(snap, signals.in_oval, hint, self._result)

    def _maybe_pad_gate(self, now_s: float) -> str | None:
        """Evaluate the live-person gate once CENTER_FACE has passed.

        Returns the occlusion hint to surface, or sets `self._result` to
        REJECT on a spoof. Returns None when the gate does not apply.
        """
        if self._pad_gate_done:
            return None
        center_passed = any(
            r.type == ChallengeType.CENTER_FACE and r.completed
            for r in self._session.results()
        )
        if not center_passed:
            return None

        self._pad_gate_done = True
        pad_verdict_value, pad_reasons = passive_verdict(self._passive_samples)
        if pad_verdict_value == Verdict.ACCEPT:
            return None  # live person — proceed to the motion challenges

        message = (
            self._last_occlusion_hint
            or (primary_message_for(pad_reasons) if pad_reasons else _NOT_LIVE_MESSAGE)
        )
        self._result = DriverResult(
            verdict=Verdict.HARD_REJECT,
            reasons=tuple(pad_reasons),
            message=message,
        )
        return self._last_occlusion_hint

    async def _maybe_sample_passive(self, snap, frame_bgr, now_s: float) -> None:
        is_stable = (
            snap.phase == SessionPhase.PREFLIGHT
            or snap.phase == SessionPhase.RECENTERING
            or (
                snap.phase == SessionPhase.CHALLENGE_ACTIVE
                and snap.current_challenge is not None
                and snap.current_challenge.type == ChallengeType.CENTER_FACE
            )
        )
        should = (
            is_stable
            and (now_s - self._last_passive_at) * 1000.0
            >= self._config.parser_throttle_ms
        )
        completed_now = sum(1 for r in snap.completed if r.completed)
        if completed_now > self._last_completed:
            should = True
        self._last_completed = completed_now

        if should:
            self._last_passive_at = now_s
            report = await self._engine.analyze_passive(frame_bgr)
            self._passive_samples.append(report)

    def _finalize(self) -> DriverResult:
        active_v, passive_v, final_v, active_reasons, passive_reasons = decide_active(
            self._session.results(), self._passive_samples
        )
        reasons = tuple(active_reasons) + tuple(passive_reasons)
        if final_v == Verdict.ACCEPT:
            message = _ACCEPT_MESSAGE
        elif any(r in _OCCLUSION_REASONS for r in reasons) and self._last_occlusion_hint:
            message = self._last_occlusion_hint
        else:
            message = primary_message_for(reasons)
        return DriverResult(verdict=final_v, reasons=reasons, message=message)
