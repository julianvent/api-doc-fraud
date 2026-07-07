"""Verdict, status, and reason taxonomies.

`Reason` codes are controlled vocabulary mapped to localized messages by
the frontend/dashboard. Adding a code is a deliberate, consumer-affecting
change.
"""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """Final outcome of liveness analysis."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    HARD_REJECT = "HARD_REJECT"


class Status(str, Enum):
    """Per-detector execution status.

    OK       — ran to completion, score meaningful.
    FAILED   — invoked but raised; result discarded by decide().
    SKIPPED  — not invoked (e.g. quality gate stopped the pipeline).
    DEGRADED — warmup failed (missing/corrupt weights, init crash).
               Pipeline runs on remaining detectors; decide() never
               ACCEPTs while any detector is DEGRADED.
    """

    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"


class Reason(str, Enum):
    """Controlled vocabulary of reasons that flow into the report.

    Hard signals (Tier 1 vetoes)
    """

    FACE_NOT_DETECTED = "face_not_detected"
    FACE_QUALITY_LOW = "face_quality_low"
    FACE_QUALITY_BLUR = "face_quality_blur"
    FACE_QUALITY_BRIGHTNESS = "face_quality_brightness"
    FACE_TOO_SMALL = "face_too_small"
    MULTIPLE_FACES = "multiple_faces"

    # Face-completeness checks (mediapipe_face.py _completeness_issues).
    # Any forces HARD_REJECT in decide(): input unusable as a sample.
    FACE_PARTIAL_FRAME = "face_partial_frame"
    FACE_LANDMARK_OUT_OF_BBOX = "face_landmark_out_of_bbox"
    FACE_GEOMETRY_INVALID = "face_geometry_invalid"

    FLIP_HARD_SCORE = "flip_hard_score"
    CDCN_HARD_SCORE = "cdcn_hard_score"
    DEEPPIXBIS_HARD_SCORE = "deeppixbis_hard_score"
    DEPTH_COLLAPSED = "depth_collapsed"
    MINIFAS_HARD_SCORE = "minifas_hard_score"
    MOIRE_PATTERN_DETECTED = "moire_pattern_detected"

    # Soft signals (Tier 2 — review candidates)
    FLIP_REVIEW = "flip_review"
    CDCN_REVIEW = "cdcn_review"
    DEEPPIXBIS_REVIEW = "deeppixbis_review"
    DEPTH_REVIEW = "depth_review"
    MINIFAS_REVIEW = "minifas_review"
    MOIRE_REVIEW = "moire_review"

    # Promotion / override
    ENSEMBLE_PROMOTION = "ensemble_promotion"
    CLEAN_OVERRIDE_APPLIED = "clean_override_applied"

    # Operational degradation
    DETECTOR_UNAVAILABLE = "detector_unavailable"

    # Active liveness (challenge) outcomes
    ACTIVE_CHALLENGE_FAILED = "active_challenge_failed"
    ACTIVE_RESPONSE_TOO_EARLY = "active_response_too_early"
    ACTIVE_FACE_LOST = "active_face_lost"
    # Round-trip motion: reached peak but didn't return to neutral.
    # Strong anti-replay signal — replays rarely match both motions.
    ACTIVE_INCOMPLETE_MOTION = "active_incomplete_motion"
    # Continuous quality gate: too many low-quality frames (blur,
    # brightness, partial face) during the challenge.
    ACTIVE_QUALITY_FAILURE = "active_quality_failure"
    # Re-centering between challenges: user didn't return to oval.
    ACTIVE_RECENTERING_FAILED = "active_recentering_failed"
    # Object (glasses/mask/hand/hair) covered the face; must be removed.
    ACTIVE_OCCLUSION = "active_occlusion"
    # Face identity changed mid-session. Anti-swap signal.
    ACTIVE_IDENTITY_MISMATCH = "active_identity_mismatch"
    # Preflight (pre-session environmental check) failures.
    PREFLIGHT_NO_FACE = "preflight_no_face"
    PREFLIGHT_LOW_QUALITY = "preflight_low_quality"
    PREFLIGHT_MULTIPLE_FACES = "preflight_multiple_faces"
    PREFLIGHT_OCCLUDED = "preflight_occluded"
    # Passive PAD lacked enough samples to vouch for the session.
    # Not a spoof signal — operational "could not verify".
    PASSIVE_INSUFFICIENT = "passive_insufficient"


# Soft reasons — reviewable signals the clean-override logic can promote
# to ACCEPT when a confirming detector reports a clean score.
SOFT_REASONS: frozenset[Reason] = frozenset(
    {
        Reason.FLIP_REVIEW,
        Reason.CDCN_REVIEW,
        Reason.DEEPPIXBIS_REVIEW,
        Reason.DEPTH_REVIEW,
        Reason.MINIFAS_REVIEW,
        Reason.MOIRE_REVIEW,
    }
)
