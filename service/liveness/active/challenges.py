"""Challenge taxonomy and the default pool.

A session starts with CENTER_FACE, then N-1 random challenges from
DEFAULT_POOL: 3D motion (TURN_LEFT/RIGHT) defeats flat/replay attacks via
parallax; expressions (BLINK, SMILE) via FaceLandmarker blendshapes. All
use round-trip detection: reach the peak AND return to neutral in-window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChallengeType(str, Enum):
    """Challenges supported by the active liveness session.

    CENTER_FACE is positioning (always first); TURN_LEFT/RIGHT are 3D
    head rotations (yaw delta); BLINK/SMILE are expressions measured via
    FaceLandmarker blendshapes (see active/landmarker.py).
    """

    CENTER_FACE = "center_face"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BLINK = "blink"
    SMILE = "smile"


@dataclass(frozen=True, slots=True)
class ChallengeSpec:
    """Static parameters of one challenge.

    Threshold semantics are type-dependent (see
    session.py::_peak_criterion_met): TURN_* uses min_yaw_delta vs
    baseline yaw_proxy; BLINK/SMILE use peak_threshold vs the blink/smile
    blendshape; CENTER_FACE uses the in_oval flag (no numeric threshold).
    """

    type: ChallengeType
    prompt: str
    window_seconds: float
    min_response_delay_s: float
    min_consecutive_frames: int
    # Used by TURN_LEFT / TURN_RIGHT only.
    min_yaw_delta: float
    # Used by BLINK / SMILE: peak intensity required (phase 0).
    peak_threshold: float
    # Used by BLINK / SMILE / TURN_*: how close to neutral the metric
    # must come back to count as "returned to baseline" (phase 1).
    return_threshold: float
    # Single-phase challenges skip the round-trip return phase.
    single_phase: bool = False


# CENTER_FACE: always the first challenge of every session.
CENTER_FACE_SPEC = ChallengeSpec(
    type=ChallengeType.CENTER_FACE,
    prompt="Center your face inside the oval",
    window_seconds=8.0,
    min_response_delay_s=0.0,
    min_consecutive_frames=10,
    min_yaw_delta=0.0,
    peak_threshold=0.0,
    return_threshold=0.0,
    single_phase=True,
)


# Directional + expression pool: random subset is chosen per session.
DEFAULT_POOL: tuple[ChallengeSpec, ...] = (
    ChallengeSpec(
        type=ChallengeType.TURN_LEFT,
        prompt="Turn your head to the LEFT",
        window_seconds=5.0,
        min_response_delay_s=0.4,
        min_consecutive_frames=3,
        # Subject turning to their LEFT shifts the nose toward IMAGE
        # RIGHT in an unmirrored selfie frame -> yaw_proxy > 0.
        min_yaw_delta=0.18,
        peak_threshold=0.0,
        return_threshold=0.08,
    ),
    ChallengeSpec(
        type=ChallengeType.TURN_RIGHT,
        prompt="Turn your head to the RIGHT",
        window_seconds=5.0,
        min_response_delay_s=0.4,
        min_consecutive_frames=3,
        min_yaw_delta=-0.18,
        peak_threshold=0.0,
        return_threshold=0.08,
    ),
    ChallengeSpec(
        type=ChallengeType.BLINK,
        prompt="Blink your eyes",
        window_seconds=5.0,
        min_response_delay_s=0.4,
        # Blinks are short; require a slightly shorter consecutive run.
        min_consecutive_frames=2,
        min_yaw_delta=0.0,
        # min(eyeBlinkLeft, eyeBlinkRight) >= 0.50 means both eyes are
        # clearly closed. Returning to "open" is < 0.20.
        peak_threshold=0.50,
        return_threshold=0.20,
    ),
    ChallengeSpec(
        type=ChallengeType.SMILE,
        prompt="Smile",
        window_seconds=5.0,
        min_response_delay_s=0.4,
        min_consecutive_frames=3,
        min_yaw_delta=0.0,
        # max(mouthSmileLeft, mouthSmileRight) >= 0.40 = clear smile.
        # Return to neutral < 0.15.
        peak_threshold=0.40,
        return_threshold=0.15,
    ),
)


POOL_BY_TYPE: dict[ChallengeType, ChallengeSpec] = {
    spec.type: spec for spec in (CENTER_FACE_SPEC,) + DEFAULT_POOL
}
