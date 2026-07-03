"""Active liveness session report contract.

Captures, under one `session_id`: the challenges requested, per-challenge
results with timing evidence, and the passive-PAD samples taken during the
session. The final verdict combines challenge completion with passive PAD.
`Reason` strings flow into top-level `reasons` for localized messaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from service.liveness.active.challenges import ChallengeType
from service.liveness.domain.report import LivenessReport
from service.liveness.domain.verdict import Verdict


@dataclass(frozen=True, slots=True)
class ChallengeResult:
    """Outcome of one challenge attempt."""

    type: ChallengeType
    completed: bool
    # Prompt-to-detection time in ms; None when never completed.
    detected_at_ms: float | None
    # Peak success-criterion value (area ratio for proximity, yaw_proxy
    # for head-turn). Aids threshold calibration without re-running.
    peak_metric: float
    # Non-completion cause, if any: "timeout" (window expired),
    # "too_early" (met before min_response_delay_s — replay-like),
    # "face_lost" (face left frame), None (succeeded).
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class ActiveSessionReport:
    """Final report of one active-liveness session."""

    session_id: str
    started_at: datetime
    elapsed_seconds: float

    # Random sequence the session asked for (in order).
    challenges_requested: tuple[ChallengeType, ...]
    challenge_results: tuple[ChallengeResult, ...]

    # Passive PAD samples (~1 frame/600ms), each a full LivenessReport.
    # Combined verdict comes from `passive_pad_verdict`.
    passive_samples: tuple[LivenessReport, ...]
    passive_pad_verdict: Verdict
    passive_pad_reasons: tuple[str, ...]

    # Combined verdict — see `decide_active()` for the policy.
    active_verdict: Verdict       # purely from challenge completion
    final_verdict: Verdict        # passive + active combined
    reasons: tuple[str, ...] = field(default_factory=tuple)
