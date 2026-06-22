"""User-facing messages mapped from `Reason` codes.

On a REJECT verdict the integrator needs a short, actionable sentence:
each `Reason` maps to one telling the user what to fix, not what failed
internally. Lives in `liveness.active` because the binary ACCEPT/REJECT
policy is part of the active contract; passive PAD reasons are mapped
here too since they surface via passive-sample aggregation.
"""

from __future__ import annotations

from service.liveness.domain.verdict import Reason


# Default message used when a Reason has no explicit mapping. Better
# than a generic "unknown" because it tells the user to retry.
DEFAULT_REJECT_MESSAGE = "Liveness check failed. Please try again with better conditions."


_USER_MESSAGES: dict[Reason, str] = {
    # Face detection / framing
    Reason.FACE_NOT_DETECTED: "We couldn't see your face. Make sure you are in front of the camera.",
    Reason.FACE_PARTIAL_FRAME: "Keep your face fully within the frame.",
    Reason.FACE_LANDMARK_OUT_OF_BBOX: "Face was detected unreliably. Look straight at the camera.",
    Reason.FACE_GEOMETRY_INVALID: "Face wasn't read correctly. Look straight at the camera with neutral expression.",
    Reason.FACE_TOO_SMALL: "Move closer to the camera so your face fills more of the frame.",
    Reason.FACE_QUALITY_LOW: "Image quality is too low. Improve lighting and hold the camera steady.",
    Reason.FACE_QUALITY_BLUR: "Image is blurry. Hold the camera steady and ensure good focus.",
    Reason.FACE_QUALITY_BRIGHTNESS: "Lighting is too dark or too bright. Adjust your surroundings.",
    Reason.MULTIPLE_FACES: "Only one person should be in the frame.",

    # Preflight
    Reason.PREFLIGHT_NO_FACE: "We couldn't see your face during the initial check.",
    Reason.PREFLIGHT_LOW_QUALITY: "Image quality is too low. Improve lighting and focus before starting.",
    Reason.PREFLIGHT_MULTIPLE_FACES: "Only one person should be visible to the camera.",
    Reason.PREFLIGHT_OCCLUDED: "Remove glasses, mask or any object from your face before starting.",

    # Active liveness
    Reason.ACTIVE_CHALLENGE_FAILED: "The action was not completed in time. Please try the check again.",
    Reason.ACTIVE_RESPONSE_TOO_EARLY: "Suspicious timing detected — please use a real, live face.",
    Reason.ACTIVE_FACE_LOST: "Your face left the frame during the action.",
    Reason.ACTIVE_INCOMPLETE_MOTION: "Motion incomplete — perform the full action and return to center.",
    Reason.ACTIVE_QUALITY_FAILURE: "Image was too unstable during the action (blur, motion, or lighting).",
    Reason.ACTIVE_RECENTERING_FAILED: "Return to the oval between actions.",
    Reason.ACTIVE_OCCLUSION: "Remove anything covering your face and try again.",
    Reason.ACTIVE_IDENTITY_MISMATCH: "A different person was detected. The same person must complete the whole check.",

    # Operational — verification could not be completed (not a spoof).
    Reason.PASSIVE_INSUFFICIENT: "We couldn't complete the verification. Please try again.",

    # Passive PAD signals
    Reason.MOIRE_PATTERN_DETECTED: "Screen reflection detected. Do not use a screen as light source.",
    Reason.MOIRE_REVIEW: "Screen-like pattern detected. Try in a different environment.",
    Reason.MINIFAS_HARD_SCORE: "Spoof check failed. Use a live, real face — not a photo or screen.",
    Reason.MINIFAS_REVIEW: "Anti-spoof signal flagged. Try again with a real face in good lighting.",
    Reason.ENSEMBLE_PROMOTION: "Multiple spoof signals detected. Use a real, live face.",

    # Operational
    Reason.DETECTOR_UNAVAILABLE: "A detector is unavailable right now. Please try again later.",
}


def user_message_for(reason: str) -> str:
    """Translate a `Reason.value` string into a user-facing sentence.

    Accepts the raw string because reports store `Reason.value` (not
    the enum). Unknown values fall back to `DEFAULT_REJECT_MESSAGE`.
    """
    try:
        member = Reason(reason)
    except ValueError:
        return DEFAULT_REJECT_MESSAGE
    return _USER_MESSAGES.get(member, DEFAULT_REJECT_MESSAGE)


def primary_message_for(reasons: tuple[str, ...]) -> str:
    """Pick the most actionable user message from a tuple of reasons.

    First matched mapping wins, so callers pass reasons in priority
    order; `decide_active` does this (active reasons before passive).
    """
    for r in reasons:
        try:
            member = Reason(r)
        except ValueError:
            continue
        if member in _USER_MESSAGES:
            return _USER_MESSAGES[member]
    return DEFAULT_REJECT_MESSAGE
