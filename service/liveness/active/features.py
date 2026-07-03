"""Per-frame geometric and expression features used by the state machine.

Extracts: yaw_proxy (normalized from BlazeFace keypoints, for TURN_*);
face_area (bbox area, kept for downstream/future use, no active challenge
consumes it yet); eye_blink/mouth_smile (FaceLandmarker blendshape
intensities for BLINK/SMILE, supplied by the caller). `extract_features()`
is the single entry point and merges the caller's blendshape kwargs with
the BlazeFace-derived metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from service.liveness.domain.face import FaceCrop, Point


_RIGHT_EYE = 0
_LEFT_EYE = 1
_NOSE_TIP = 2


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """Geometric + expression features extracted from one analyzed frame.

    Fields are `None` when the underlying detection did not produce
    them — callers must handle this (the session keeps `metric=0.0`
    fallbacks where appropriate).
    """

    face_area: float | None
    yaw_proxy: float | None
    eye_blink: float | None
    mouth_smile: float | None


def extract_features(
    face: FaceCrop | None,
    *,
    eye_blink: float | None = None,
    mouth_smile: float | None = None,
) -> FrameFeatures:
    """Pull geometric features from `face` and merge external blendshapes.

    `eye_blink` is min(eyeBlinkLeft, eyeBlinkRight) (both eyes closed);
    `mouth_smile` is max(mouthSmileLeft, mouthSmileRight) (either corner).
    See `active/landmarker.py::FaceBlendshapes`.
    """
    if face is None:
        return FrameFeatures(
            face_area=None,
            yaw_proxy=None,
            eye_blink=eye_blink,
            mouth_smile=mouth_smile,
        )

    return FrameFeatures(
        face_area=float(face.bbox.area),
        yaw_proxy=_estimate_yaw_proxy(face.landmarks),
        eye_blink=eye_blink,
        mouth_smile=mouth_smile,
    )


def _estimate_yaw_proxy(landmarks: tuple[Point, ...]) -> float | None:
    """Yaw proxy from the offset of nose vs the eye midpoint.

    Sign convention (unmirrored frame):
      yaw_proxy > 0  : subject turned to their LEFT
      yaw_proxy < 0  : subject turned to their RIGHT
    """
    if len(landmarks) <= max(_RIGHT_EYE, _LEFT_EYE, _NOSE_TIP):
        return None
    right_eye = landmarks[_RIGHT_EYE]
    left_eye = landmarks[_LEFT_EYE]
    nose = landmarks[_NOSE_TIP]
    eye_midpoint_x = (right_eye.x + left_eye.x) / 2.0
    eye_dist = abs(left_eye.x - right_eye.x)
    if eye_dist < 1.0:
        return None
    return float((nose.x - eye_midpoint_x) / eye_dist)
