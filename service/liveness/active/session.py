"""ActiveSession — frame-by-frame state machine for the challenge loop.

Flow: PENDING -start()-> PREFLIGHT (~2 s environment check) ->
CHALLENGE_ACTIVE(CENTER_FACE) (always first; captures baseline_area,
baseline_yaw, identity) -> CHALLENGE_ACTIVE(motion) (two-phase round-trip;
a recoverable failure goes through CHALLENGE_RETRY once) -> RECENTERING
(return to oval; identity re-check) -> next motion -> COMPLETED/FAILED.

Key behaviours: PREFLIGHT validates the environment; motion challenges are
directional (TURN_LEFT/RIGHT) or expression-based (BLINK, SMILE) and the
prompt commits to one; `wrong_direction` flags moving opposite to the
requested turn in phase 0; round-trip detection (peak AND return to neutral
in-window) defeats forward-only replay; CHALLENGE_RETRY grants one retry on
a recoverable failure; RECENTERING re-checks identity vs CENTER_FACE via the
caller's embedding; >30 % bad-quality frames -> ACTIVE_QUALITY_FAILURE;
occlusion blocks progress in every phase.

The PAD "live person" gate lives in ActiveSessionDriver, not here — this
core only handles challenges, quality, occlusion and identity. The driver
samples passive PAD and evaluates the gate once CENTER_FACE passes.

The session is FED frames and EMITS snapshots; it does NOT own the camera,
screen, oval geometry, or PAD/embedding models — those live in the
driver/engine.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from enum import Enum

import numpy as np

from service.liveness.active.challenges import (
    CENTER_FACE_SPEC,
    DEFAULT_POOL,
    ChallengeSpec,
    ChallengeType,
)
from service.liveness.active.face_embedding import is_same_identity
from service.liveness.active.features import FrameFeatures, extract_features
from service.liveness.domain.active_report import ChallengeResult
from service.liveness.domain.face import FaceCrop
from service.liveness.infrastructure.logging import get_logger, request_logger


_log = get_logger(__name__)


class SessionPhase(str, Enum):
    """High-level state of a session."""

    PENDING = "pending"
    PREFLIGHT = "preflight"
    CHALLENGE_ACTIVE = "challenge_active"
    CHALLENGE_RETRY = "challenge_retry"
    RECENTERING = "recentering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable projection of session state at one moment (HUD render).

    Mutable state lives inside `ActiveSession`.
    """

    phase: SessionPhase
    current_challenge: ChallengeSpec | None
    challenge_index: int        # 0-based, of n_challenges
    n_challenges: int
    elapsed_in_phase_s: float
    time_remaining_s: float     # of the current challenge window; 0 outside
    current_metric: float       # peak-phase metric; meaning depends on type
    challenge_phase: int        # 0 = peak, 1 = return; 0 for CENTER_FACE
    wrong_direction: bool       # user is moving opposite to the requested direction
    completed: tuple[ChallengeResult, ...]
    # Preflight progress (only meaningful while phase == PREFLIGHT).
    # `*_duration_s` defaults mirror the tunables below and are also set
    # explicitly per snapshot for clarity in the JSON report.
    preflight_elapsed_s: float = 0.0
    preflight_duration_s: float = 2.0
    # Retry context (only meaningful while phase == CHALLENGE_RETRY).
    retry_for_challenge: ChallengeSpec | None = None
    retry_previous_reason: str | None = None
    retry_elapsed_s: float = 0.0
    retry_duration_s: float = 2.0


# Tunables.
_FACE_LOST_TIMEOUT_S = 1.0
_RECENTER_TIMEOUT_S = 3.5
_RECENTER_MIN_HOLD_S = 0.4
# Fraction of bad-quality frames allowed during a single challenge.
_BAD_QUALITY_FRACTION_LIMIT = 0.30
# Magnitude of the "wrong direction" hint trigger (only in phase 0).
_WRONG_DIR_AREA_MARGIN = 0.15
_WRONG_DIR_YAW_MARGIN = 0.10
# Preflight observation window.
_PREFLIGHT_DURATION_S = 2.0
# Pass threshold: fraction of preflight frames with a detected face AND
# acceptable quality.
_PREFLIGHT_PASS_FRACTION = 0.60
# Brief transition before re-running a failed challenge.
_RETRY_HOLD_S = 2.0
# Failure reasons granting ONE retry per challenge; others fail
# immediately. `too_early` is excluded (anti-replay signal, not user
# error); `occluded` is retryable (remove the object and try again).
_RETRYABLE_FAILURE_REASONS: frozenset[str] = frozenset(
    {"timeout", "face_lost", "incomplete_motion", "quality_failure", "occluded"}
)
# Fraction of occluded frames to attribute a failure to occlusion (vs
# timeout). Also the preflight occlusion-fail threshold.
_OCCLUSION_FRACTION_LIMIT = 0.30


class ActiveSession:
    """Drive a sequence of challenges from streamed frames.

    Usage:
        s = ActiveSession(n_challenges=3, seed=42)
        s.start(t0)
        for each frame:
            snap = s.submit(face, t, in_oval=oval_check)
            if snap.phase in (COMPLETED, FAILED): break

    `n_challenges` is the TOTAL including the always-first CENTER_FACE, so
    n_challenges=3 means CENTER_FACE + 2 random directional ones.
    """

    def __init__(
        self,
        *,
        n_challenges: int = 3,
        pool: tuple[ChallengeSpec, ...] = DEFAULT_POOL,
        seed: int | None = None,
    ) -> None:
        if n_challenges < 1:
            raise ValueError("n_challenges must be >= 1")
        # n_challenges includes CENTER_FACE; the rest are drawn from `pool`.
        n_random = n_challenges - 1
        if n_random > len(pool):
            raise ValueError(
                f"n_challenges={n_challenges} (-1 for CENTER_FACE) "
                f"exceeds directional pool size={len(pool)}"
            )
        rng = random.Random(seed) if seed is not None else random.SystemRandom()
        directional = tuple(rng.sample(pool, k=n_random)) if n_random > 0 else ()
        self._selected: tuple[ChallengeSpec, ...] = (CENTER_FACE_SPEC,) + directional

        self._n_challenges = n_challenges
        self._session_id = str(uuid.uuid4())
        self._log: logging.LoggerAdapter = request_logger(_log, self._session_id)

        # Phase / timing.
        self._phase = SessionPhase.PENDING
        self._phase_started_at: float | None = None
        self._challenge_idx = 0
        self._current_prompt_at: float | None = None

        # Per-challenge state — reset by `_enter_challenge`.
        self._challenge_phase: int = 0  # 0 = peak, 1 = return
        self._consecutive_hits: int = 0
        self._peak_metric: float = 0.0
        self._peak_reached_at: float | None = None
        self._wrong_direction: bool = False
        self._challenge_total_frames: int = 0
        self._challenge_bad_quality: int = 0
        self._challenge_occluded_frames: int = 0

        self._last_face_seen_at: float | None = None
        self._recenter_in_oval_streak_s: float = 0.0
        self._last_t_in_recentering: float | None = None

        # Baseline (captured the moment CENTER_FACE passes).
        self._baseline_area: float | None = None
        self._baseline_yaw: float | None = None
        # Identity embedding captured at CENTER_FACE; later frames in
        # neutral pose (RECENTERING) are checked against it.
        self._identity_baseline: np.ndarray | None = None

        # Retry tracking: each entry is a challenge index that has
        # already used its single retry attempt.
        self._retry_used: set[int] = set()
        self._retry_started_at: float | None = None
        self._retry_for_idx: int | None = None
        self._retry_failure_reason: str | None = None

        # Preflight counters (reset on start()).
        self._preflight_total_frames = 0
        self._preflight_face_frames = 0
        self._preflight_quality_ok_frames = 0
        self._preflight_multi_face_frames = 0
        self._preflight_occluded_frames = 0

        self._results: list[ChallengeResult] = []

    # ------------------------------------------------------------------
    # Introspection.
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def selected_challenges(self) -> tuple[ChallengeType, ...]:
        return tuple(spec.type for spec in self._selected)

    @property
    def is_terminal(self) -> bool:
        return self._phase in (SessionPhase.COMPLETED, SessionPhase.FAILED)

    def results(self) -> tuple[ChallengeResult, ...]:
        return tuple(self._results)

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def start(self, now_s: float) -> None:
        """Mark the session as started. Enters PREFLIGHT first."""
        if self._phase != SessionPhase.PENDING:
            raise RuntimeError("session already started")
        self._phase = SessionPhase.PREFLIGHT
        self._phase_started_at = now_s
        self._preflight_total_frames = 0
        self._preflight_face_frames = 0
        self._preflight_quality_ok_frames = 0
        self._preflight_multi_face_frames = 0
        self._preflight_occluded_frames = 0
        self._log.info(
            "active.session_started",
            extra={
                "challenges": [s.type.value for s in self._selected],
                "n_challenges": self._n_challenges,
            },
        )

    def submit(
        self,
        face: FaceCrop | None,
        now_s: float,
        *,
        in_oval: bool = False,
        n_faces_detected: int = 1,
        eye_blink: float | None = None,
        mouth_smile: float | None = None,
        occluded: bool = False,
        embedding: np.ndarray | None = None,
    ) -> SessionSnapshot:
        """Process one frame and return the resulting snapshot.

        Optional signal kwargs:
          - `n_faces_detected`: preflight fails on `multiple_faces` if > 1.
          - `eye_blink`: min(eyeBlinkLeft, eyeBlinkRight), 1.0 = closed.
          - `mouth_smile`: max(mouthSmileLeft, mouthSmileRight), 1.0 = strong.
          - `occluded`: object covers the face (caller's check). Blocks
            progress everywhere and attributes failures to occlusion so the
            user sees "remove the object" instead of a generic timeout.
          - `embedding`: L2-normalized frame embedding (FaceEmbedder).
            Captured as the identity baseline when CENTER_FACE passes and
            re-checked in neutral pose (RECENTERING) to catch a person swap.
        """
        if self._phase == SessionPhase.PENDING:
            raise RuntimeError("call start() before submit()")
        if self.is_terminal:
            return self._snapshot(now_s, current_metric=0.0)

        features = extract_features(face, eye_blink=eye_blink, mouth_smile=mouth_smile)
        if face is not None and features.face_area is not None:
            self._last_face_seen_at = now_s

        if self._phase == SessionPhase.PREFLIGHT:
            return self._step_preflight(face, n_faces_detected, occluded, now_s)
        if self._phase == SessionPhase.CHALLENGE_ACTIVE:
            return self._step_challenge(
                face, features, in_oval, occluded, embedding, now_s
            )
        if self._phase == SessionPhase.CHALLENGE_RETRY:
            return self._step_retry(now_s)
        if self._phase == SessionPhase.RECENTERING:
            return self._step_recentering(in_oval, occluded, embedding, now_s)
        return self._snapshot(now_s, current_metric=0.0)

    # ------------------------------------------------------------------
    # PREFLIGHT step.
    # ------------------------------------------------------------------

    def _step_preflight(
        self,
        face: FaceCrop | None,
        n_faces_detected: int,
        occluded: bool,
        now_s: float,
    ) -> SessionSnapshot:
        """Observe the input for ~2 s before challenges begin.

        Pass requires the MAJORITY of frames had a single, unoccluded,
        detected face with acceptable quality. Failure attribution prefers
        the dominant cause for a clear hint.
        """
        self._preflight_total_frames += 1
        if face is not None:
            self._preflight_face_frames += 1
            if face.quality.is_acceptable:
                self._preflight_quality_ok_frames += 1
        if n_faces_detected > 1:
            self._preflight_multi_face_frames += 1
        if occluded:
            self._preflight_occluded_frames += 1

        elapsed = now_s - self._phase_t0_or(now_s)
        if elapsed < _PREFLIGHT_DURATION_S:
            return self._snapshot(now_s, current_metric=0.0)

        total = max(1, self._preflight_total_frames)
        face_frac = self._preflight_face_frames / total
        quality_frac = self._preflight_quality_ok_frames / total
        multi_frac = self._preflight_multi_face_frames / total
        occluded_frac = self._preflight_occluded_frames / total

        # Priority: multi-face (security) > no-face (nothing else assessable)
        # > occlusion (actionable) > low-quality (catch-all).
        if multi_frac > _OCCLUSION_FRACTION_LIMIT:
            self._fail_preflight(now_s, "multiple_faces", face_frac, quality_frac, multi_frac, occluded_frac)
            return self._snapshot(now_s, current_metric=0.0)
        if face_frac < _PREFLIGHT_PASS_FRACTION:
            self._fail_preflight(now_s, "no_face", face_frac, quality_frac, multi_frac, occluded_frac)
            return self._snapshot(now_s, current_metric=0.0)
        if occluded_frac > _OCCLUSION_FRACTION_LIMIT:
            self._fail_preflight(now_s, "occluded", face_frac, quality_frac, multi_frac, occluded_frac)
            return self._snapshot(now_s, current_metric=0.0)
        if quality_frac < _PREFLIGHT_PASS_FRACTION:
            self._fail_preflight(now_s, "low_quality", face_frac, quality_frac, multi_frac, occluded_frac)
            return self._snapshot(now_s, current_metric=0.0)

        # Passed preflight — enter the first challenge.
        self._log.info(
            "active.preflight_passed",
            extra={
                "face_frac": round(face_frac, 2),
                "quality_frac": round(quality_frac, 2),
                "occluded_frac": round(occluded_frac, 2),
                "total_frames": self._preflight_total_frames,
            },
        )
        self._enter_challenge(0, now_s)
        return self._snapshot(now_s, current_metric=0.0)

    def _fail_preflight(
        self,
        now_s: float,
        kind: str,
        face_frac: float,
        quality_frac: float,
        multi_frac: float,
        occluded_frac: float,
    ) -> None:
        """Mark the session FAILED with a synthetic preflight result.

        The synthetic ChallengeResult lets `decide_active` attribute the
        REJECT to a specific preflight reason, not "no challenges ran".
        """
        reason_map = {
            "no_face": "preflight_no_face",
            "low_quality": "preflight_low_quality",
            "multiple_faces": "preflight_multiple_faces",
            "occluded": "preflight_occluded",
        }
        self._results.append(
            ChallengeResult(
                type=ChallengeType.CENTER_FACE,  # closest semantic match
                completed=False,
                detected_at_ms=None,
                peak_metric=0.0,
                failure_reason=reason_map[kind],
            )
        )
        self._phase = SessionPhase.FAILED
        self._log.info(
            "active.preflight_failed",
            extra={
                "kind": kind,
                "face_frac": round(face_frac, 2),
                "quality_frac": round(quality_frac, 2),
                "multi_frac": round(multi_frac, 2),
                "occluded_frac": round(occluded_frac, 2),
                "total_frames": self._preflight_total_frames,
            },
        )

    # ------------------------------------------------------------------
    # CHALLENGE_RETRY step.
    # ------------------------------------------------------------------

    def _step_retry(self, now_s: float) -> SessionSnapshot:
        """Brief transition phase: show "Try again" then re-enter the challenge."""
        elapsed = now_s - (self._retry_started_at or now_s)
        if elapsed >= _RETRY_HOLD_S:
            idx = self._retry_for_idx
            assert idx is not None
            self._retry_for_idx = None
            self._retry_started_at = None
            self._retry_failure_reason = None
            self._enter_challenge(idx, now_s)
        return self._snapshot(now_s, current_metric=0.0)

    # ------------------------------------------------------------------
    # CHALLENGE_ACTIVE step.
    # ------------------------------------------------------------------

    def _step_challenge(
        self,
        face: FaceCrop | None,
        features: FrameFeatures,
        in_oval: bool,
        occluded: bool,
        embedding: "np.ndarray | None",
        now_s: float,
    ) -> SessionSnapshot:
        spec = self._selected[self._challenge_idx]
        prompt_at = self._prompt_at_or(now_s)
        since_prompt = now_s - prompt_at

        # Continuous quality gate (Block A.6).
        self._challenge_total_frames += 1
        if face is not None and not face.quality.is_acceptable:
            self._challenge_bad_quality += 1
        if occluded:
            self._challenge_occluded_frames += 1

        # Face lost (after grace period).
        if since_prompt > 0.3 and (
            self._last_face_seen_at is None
            or now_s - self._last_face_seen_at > _FACE_LOST_TIMEOUT_S
        ):
            self._fail_challenge(spec, now_s, "face_lost")
            return self._snapshot(now_s, current_metric=0.0)

        # Occlusion blocks ALL progress: resets the hit streak so the
        # criterion can't accumulate; if it persists to timeout the failure
        # is attributed to occlusion (see timeout block).
        if occluded:
            self._consecutive_hits = 0

        # ----- CENTER_FACE (single-phase, in_oval criterion) -------------
        if spec.type == ChallengeType.CENTER_FACE:
            if in_oval and not occluded:
                self._consecutive_hits += 1
                if self._consecutive_hits >= spec.min_consecutive_frames:
                    # Capture baselines at the moment we PASS the centring.
                    if features.face_area is not None:
                        self._baseline_area = features.face_area
                    if features.yaw_proxy is not None:
                        self._baseline_yaw = features.yaw_proxy
                    if embedding is not None:
                        self._identity_baseline = embedding
                        self._log.info("active.identity_baseline_captured")
                    self._pass_challenge(spec, now_s, since_prompt, current_metric=0.0)
                    return self._snapshot(now_s, current_metric=0.0)
            else:
                self._consecutive_hits = 0

            if since_prompt > spec.window_seconds:
                self._fail_challenge(spec, now_s, self._timeout_reason("timeout"))
            return self._snapshot(now_s, current_metric=0.0)

        # ----- Motion challenges (two-phase round-trip) ------------------
        metric = self._metric_for(spec, features)

        # Wrong-direction detection (only meaningful in phase 0).
        if self._challenge_phase == 0:
            self._wrong_direction = _is_wrong_direction(spec, metric)

        if self._challenge_phase == 0:
            self._peak_metric = _peak(self._peak_metric, metric, spec)
            if _peak_criterion_met(spec, metric):
                self._consecutive_hits += 1
                if self._consecutive_hits >= spec.min_consecutive_frames:
                    if since_prompt < spec.min_response_delay_s:
                        # Suspiciously fast = pre-recorded.
                        self._fail_challenge(spec, now_s, "too_early")
                        return self._snapshot(now_s, current_metric=metric)
                    self._challenge_phase = 1
                    self._consecutive_hits = 0
                    self._peak_reached_at = now_s
                    self._log.info(
                        "active.peak_reached",
                        extra={
                            "challenge_type": spec.type.value,
                            "peak_metric": round(self._peak_metric, 3),
                            "since_prompt_s": round(since_prompt, 2),
                        },
                    )
            else:
                self._consecutive_hits = 0
        else:
            # Phase 1: looking for return to baseline.
            if _return_criterion_met(spec, metric):
                self._consecutive_hits += 1
                if self._consecutive_hits >= spec.min_consecutive_frames:
                    # Both phases done within window. Check quality gate.
                    if self._quality_gate_failed():
                        self._fail_challenge(spec, now_s, "quality_failure")
                    else:
                        self._pass_challenge(
                            spec, now_s, since_prompt, current_metric=metric
                        )
                    return self._snapshot(now_s, current_metric=metric)
            else:
                self._consecutive_hits = 0

        # Window timeout.
        if since_prompt > spec.window_seconds:
            # Distinguish "never reached peak" from "peak but no return".
            base_reason = "timeout" if self._challenge_phase == 0 else "incomplete_motion"
            self._fail_challenge(spec, now_s, self._timeout_reason(base_reason))
            return self._snapshot(now_s, current_metric=metric)

        return self._snapshot(now_s, current_metric=metric)

    # ------------------------------------------------------------------
    # RECENTERING step.
    # ------------------------------------------------------------------

    def _step_recentering(
        self, in_oval: bool, occluded: bool,
        embedding: "np.ndarray | None", now_s: float,
    ) -> SessionSnapshot:
        phase_t0 = self._phase_t0_or(now_s)
        elapsed = now_s - phase_t0

        if self._last_t_in_recentering is not None:
            dt = max(0.0, now_s - self._last_t_in_recentering)
        else:
            dt = 0.0
        self._last_t_in_recentering = now_s

        # Identity check — RECENTERING's neutral pose confirms it's still
        # the same person. `is_same_identity` returns True when either
        # embedding is absent, so a missing frame does not falsely reject.
        if (
            in_oval
            and not occluded
            and self._identity_baseline is not None
            and embedding is not None
            and not is_same_identity(self._identity_baseline, embedding)
        ):
            self._results.append(
                ChallengeResult(
                    type=ChallengeType.CENTER_FACE,
                    completed=False,
                    detected_at_ms=None,
                    peak_metric=0.0,
                    failure_reason="identity_mismatch",
                )
            )
            self._phase = SessionPhase.FAILED
            self._log.info("active.identity_mismatch")
            return self._snapshot(now_s, current_metric=0.0)

        # An occluded face is not "centered" — block the recenter streak.
        if in_oval and not occluded:
            self._recenter_in_oval_streak_s += dt
        else:
            self._recenter_in_oval_streak_s = 0.0

        if self._recenter_in_oval_streak_s >= _RECENTER_MIN_HOLD_S:
            next_idx = self._challenge_idx + 1
            if next_idx >= self._n_challenges:
                self._phase = SessionPhase.COMPLETED
                self._log.info("active.session_completed", extra={"all_passed": True})
            else:
                self._enter_challenge(next_idx, now_s)
            return self._snapshot(now_s, current_metric=0.0)

        if elapsed > _RECENTER_TIMEOUT_S:
            # Synthesize a failed result so the report attributes failure to
            # the recentering step, not "no reason".
            self._results.append(
                ChallengeResult(
                    type=ChallengeType.CENTER_FACE,
                    completed=False,
                    detected_at_ms=None,
                    peak_metric=0.0,
                    failure_reason="recentering_failed",
                )
            )
            self._phase = SessionPhase.FAILED
            self._log.info(
                "active.recentering_failed",
                extra={"elapsed_s": round(elapsed, 2)},
            )

        return self._snapshot(now_s, current_metric=0.0)

    # ------------------------------------------------------------------
    # Challenge outcomes.
    # ------------------------------------------------------------------

    def _enter_challenge(self, idx: int, now_s: float) -> None:
        self._phase = SessionPhase.CHALLENGE_ACTIVE
        self._phase_started_at = now_s
        self._challenge_idx = idx
        self._current_prompt_at = now_s
        # Reset per-challenge state.
        self._challenge_phase = 0
        self._consecutive_hits = 0
        self._peak_metric = 0.0
        self._peak_reached_at = None
        self._wrong_direction = False
        self._challenge_total_frames = 0
        self._challenge_bad_quality = 0
        self._challenge_occluded_frames = 0
        # Reset recentering trackers (used by the *next* RECENTERING phase).
        self._recenter_in_oval_streak_s = 0.0
        self._last_t_in_recentering = None

        spec = self._selected[idx]
        self._log.info(
            "active.challenge_prompt",
            extra={
                "challenge_idx": idx,
                "challenge_type": spec.type.value,
                "window_seconds": spec.window_seconds,
            },
        )

    def _pass_challenge(
        self,
        spec: ChallengeSpec,
        now_s: float,
        since_prompt_s: float,
        current_metric: float,
    ) -> None:
        if self._quality_gate_failed():
            # Quality issues during the challenge — overrule the pass.
            self._fail_challenge(spec, now_s, "quality_failure")
            return
        result = ChallengeResult(
            type=spec.type,
            completed=True,
            detected_at_ms=since_prompt_s * 1000.0,
            peak_metric=self._peak_metric or current_metric,
            failure_reason=None,
        )
        self._results.append(result)
        self._log.info(
            "active.challenge_passed",
            extra={
                "challenge_type": spec.type.value,
                "detected_at_ms": round(result.detected_at_ms, 1),
                "peak_metric": round(result.peak_metric, 3),
            },
        )
        next_idx = self._challenge_idx + 1
        if next_idx >= self._n_challenges:
            self._phase = SessionPhase.COMPLETED
            self._log.info("active.session_completed", extra={"all_passed": True})
            return
        # Skip RECENTERING after CENTER_FACE — already centred.
        if spec.type == ChallengeType.CENTER_FACE:
            self._enter_challenge(next_idx, now_s)
        else:
            self._phase = SessionPhase.RECENTERING
            self._phase_started_at = now_s
            self._current_prompt_at = None
            self._recenter_in_oval_streak_s = 0.0
            self._last_t_in_recentering = None
            self._log.info(
                "active.recentering_started",
                extra={"after_challenge": spec.type.value},
            )

    def _fail_challenge(
        self, spec: ChallengeSpec, now_s: float, reason: str
    ) -> None:
        retry_eligible = (
            reason in _RETRYABLE_FAILURE_REASONS
            and self._challenge_idx not in self._retry_used
        )

        self._log.info(
            "active.challenge_failed",
            extra={
                "challenge_type": spec.type.value,
                "reason": reason,
                "peak_metric": round(self._peak_metric, 3),
                "challenge_phase": self._challenge_phase,
                "bad_quality_frames": self._challenge_bad_quality,
                "total_frames": self._challenge_total_frames,
                "retry_eligible": retry_eligible,
            },
        )

        if retry_eligible:
            # Defer recording the result until the retry also fails.
            self._retry_used.add(self._challenge_idx)
            self._retry_for_idx = self._challenge_idx
            self._retry_started_at = now_s
            self._retry_failure_reason = reason
            self._phase = SessionPhase.CHALLENGE_RETRY
            self._phase_started_at = now_s
            return

        # Final failure (no retry left, or non-retryable reason).
        result = ChallengeResult(
            type=spec.type,
            completed=False,
            detected_at_ms=None,
            peak_metric=self._peak_metric,
            failure_reason=reason,
        )
        self._results.append(result)
        self._phase = SessionPhase.FAILED

    # ------------------------------------------------------------------
    # Snapshot building.
    # ------------------------------------------------------------------

    def _snapshot(self, now_s: float, current_metric: float) -> SessionSnapshot:
        elapsed = now_s - self._phase_t0_or(now_s)
        spec = (
            self._selected[self._challenge_idx]
            if self._phase == SessionPhase.CHALLENGE_ACTIVE
            and self._challenge_idx < len(self._selected)
            else None
        )
        time_remaining = 0.0
        if spec is not None and self._current_prompt_at is not None:
            time_remaining = max(
                0.0,
                spec.window_seconds - (now_s - self._current_prompt_at),
            )

        preflight_elapsed = (
            elapsed if self._phase == SessionPhase.PREFLIGHT else 0.0
        )
        retry_spec: ChallengeSpec | None = None
        retry_elapsed = 0.0
        retry_reason: str | None = None
        if self._phase == SessionPhase.CHALLENGE_RETRY and self._retry_for_idx is not None:
            retry_spec = self._selected[self._retry_for_idx]
            retry_elapsed = now_s - (self._retry_started_at or now_s)
            retry_reason = self._retry_failure_reason

        return SessionSnapshot(
            phase=self._phase,
            current_challenge=spec,
            challenge_index=self._challenge_idx,
            n_challenges=self._n_challenges,
            elapsed_in_phase_s=elapsed,
            time_remaining_s=time_remaining,
            current_metric=current_metric,
            challenge_phase=self._challenge_phase,
            wrong_direction=self._wrong_direction,
            completed=tuple(self._results),
            preflight_elapsed_s=preflight_elapsed,
            preflight_duration_s=_PREFLIGHT_DURATION_S,
            retry_for_challenge=retry_spec,
            retry_previous_reason=retry_reason,
            retry_elapsed_s=retry_elapsed,
            retry_duration_s=_RETRY_HOLD_S,
        )

    # ------------------------------------------------------------------
    # Metric and quality helpers.
    # ------------------------------------------------------------------

    def _metric_for(
        self, spec: ChallengeSpec, features: FrameFeatures
    ) -> float:
        """Return the signed metric the criterion threshold compares.

        TURN_*: baseline-relative yaw delta. BLINK/SMILE: raw intensity
        (no baseline — neutral is 0).
        """
        if spec.type in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT):
            if features.yaw_proxy is None or self._baseline_yaw is None:
                return 0.0
            return features.yaw_proxy - self._baseline_yaw
        if spec.type == ChallengeType.BLINK:
            return features.eye_blink if features.eye_blink is not None else 0.0
        if spec.type == ChallengeType.SMILE:
            return features.mouth_smile if features.mouth_smile is not None else 0.0
        return 0.0

    def _quality_gate_failed(self) -> bool:
        if self._challenge_total_frames < 5:
            # Too few frames to judge — let the result stand.
            return False
        return (
            self._challenge_bad_quality / self._challenge_total_frames
            > _BAD_QUALITY_FRACTION_LIMIT
        )

    def _timeout_reason(self, base_reason: str) -> str:
        """Attribute a timeout to occlusion when an object dominated the
        challenge, else return base_reason (timeout / incomplete_motion).
        Makes the reject say "remove the object" instead of "not completed
        in time" when the real cause was a covered face."""
        if self._challenge_total_frames >= 5 and (
            self._challenge_occluded_frames / self._challenge_total_frames
            > _OCCLUSION_FRACTION_LIMIT
        ):
            return "occluded"
        return base_reason

    def _phase_t0_or(self, fallback: float) -> float:
        return (
            self._phase_started_at if self._phase_started_at is not None else fallback
        )

    def _prompt_at_or(self, fallback: float) -> float:
        return (
            self._current_prompt_at if self._current_prompt_at is not None else fallback
        )


# ----------------------------------------------------------------------
# Pure helpers — separated so they are unit-testable.
# ----------------------------------------------------------------------


def _peak_criterion_met(spec: ChallengeSpec, metric: float) -> bool:
    """Threshold check for the PEAK phase of one challenge."""
    if spec.type == ChallengeType.TURN_LEFT:
        return metric >= spec.min_yaw_delta
    if spec.type == ChallengeType.TURN_RIGHT:
        return metric <= spec.min_yaw_delta
    if spec.type in (ChallengeType.BLINK, ChallengeType.SMILE):
        return metric >= spec.peak_threshold
    return False


def _return_criterion_met(spec: ChallengeSpec, metric: float) -> bool:
    """Threshold check for the RETURN phase (back-to-neutral)."""
    if spec.type in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT):
        return abs(metric) <= spec.return_threshold
    if spec.type in (ChallengeType.BLINK, ChallengeType.SMILE):
        # Neutral for blink/smile is 0 (eyes open / mouth neutral).
        return metric <= spec.return_threshold
    return False


def _peak(prev_peak: float, metric: float, spec: ChallengeSpec) -> float:
    """Track the most extreme value seen so far for this challenge."""
    if spec.type == ChallengeType.TURN_LEFT:
        return max(prev_peak, metric)
    if spec.type == ChallengeType.TURN_RIGHT:
        if prev_peak == 0.0:
            return metric
        return min(prev_peak, metric)
    if spec.type in (ChallengeType.BLINK, ChallengeType.SMILE):
        return max(prev_peak, metric)
    return prev_peak


def _is_wrong_direction(spec: ChallengeSpec, metric: float) -> bool:
    """Detect motion opposite to the requested direction.

    Only meaningful in phase 0 (peak chase); N/A for BLINK/SMILE (no
    opposite expression).
    """
    if spec.type == ChallengeType.TURN_LEFT:
        return metric < -_WRONG_DIR_YAW_MARGIN
    if spec.type == ChallengeType.TURN_RIGHT:
        return metric > _WRONG_DIR_YAW_MARGIN
    return False
