"""MediaPipe FaceLandmarker wrapper for blink / smile detection.

Thin adapter over the MediaPipe Tasks API, exposing only what active
liveness needs: per-frame eye/mouth blendshape intensities from the
strongest detected face. Kept separate from the localizer (BlazeFace) so
the 15-25 ms landmark cost isn't paid on every passive `analyze()` call —
it only runs during an active session.

Blendshapes returned (subset of ~52): eyeBlinkLeft/Right (0=open,
1=closed) and mouthSmileLeft/Right (0=neutral, 1=smile peak).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from service.liveness.exceptions import (
    ModelLoadError,
    WeightsHashMismatch,
    WeightsNotFound,
)
from service.liveness.infrastructure.loader import resolve as resolve_weight
from service.liveness.infrastructure.logging import get_logger


_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FaceBlendshapes:
    """Subset of FaceLandmarker blendshapes used for active challenges.

    Each field is in [0, 1]. `None` indicates the underlying detection
    did not produce blendshapes (no face, or the model output was empty).
    """

    eye_blink_left: float | None
    eye_blink_right: float | None
    mouth_smile_left: float | None
    mouth_smile_right: float | None

    @property
    def blink_intensity(self) -> float | None:
        """`min(left, right)` — both eyes must be closed to count as a blink."""
        if self.eye_blink_left is None or self.eye_blink_right is None:
            return None
        return min(self.eye_blink_left, self.eye_blink_right)

    @property
    def smile_intensity(self) -> float | None:
        """`max(left, right)` — either corner smiling is enough."""
        if self.mouth_smile_left is None or self.mouth_smile_right is None:
            return None
        return max(self.mouth_smile_left, self.mouth_smile_right)


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    """Output of one `extract()` call.

    `landmarks_xy` is an (N, 2) pixel-coord array (478 rows), or `None`
    when no face was detected — the same condition that yields empty
    `blendshapes`.
    """

    blendshapes: "FaceBlendshapes"
    landmarks_xy: np.ndarray | None


_EMPTY_BS = FaceBlendshapes(None, None, None, None)
_EMPTY_RESULT = LandmarkResult(blendshapes=_EMPTY_BS, landmarks_xy=None)


class MediaPipeFaceLandmarker:
    """Thin wrapper exposing `extract(image)` only."""

    name = "mediapipe_face_landmarker"

    def __init__(
        self,
        *,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
    ) -> None:
        self._min_detection_confidence = min_face_detection_confidence
        self._min_presence_confidence = min_face_presence_confidence
        self._landmarker: vision.FaceLandmarker | None = None

    def warmup(self) -> None:
        if self._landmarker is not None:
            return
        try:
            model_path: Path = resolve_weight("face_landmarker")
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=self._min_detection_confidence,
                min_face_presence_confidence=self._min_presence_confidence,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except (WeightsNotFound, WeightsHashMismatch):
            raise
        except Exception as exc:  # noqa: BLE001 — wrap framework errors
            _log.exception("face_landmarker.warmup_failed", extra={"localizer": self.name})
            raise ModelLoadError(
                f"FaceLandmarker failed to warm up: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("face_landmarker.ready", extra={"localizer": self.name})

    def extract(self, image_bgr: np.ndarray) -> LandmarkResult:
        """Return blendshapes and landmark pixel coords for the strongest face.

        Returns `LandmarkResult` with empty fields when no face is detected.
        """
        if self._landmarker is None:
            self.warmup()
        assert self._landmarker is not None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            _log.exception("face_landmarker.detect_failed")
            return _EMPTY_RESULT

        if not result.face_blendshapes:
            return _EMPTY_RESULT

        # FaceLandmarker returns a list per detected face; we requested
        # num_faces=1, so the only entry is at [0].
        scores = {bs.category_name: float(bs.score) for bs in result.face_blendshapes[0]}
        blendshapes = FaceBlendshapes(
            eye_blink_left=scores.get("eyeBlinkLeft"),
            eye_blink_right=scores.get("eyeBlinkRight"),
            mouth_smile_left=scores.get("mouthSmileLeft"),
            mouth_smile_right=scores.get("mouthSmileRight"),
        )

        landmarks_xy: np.ndarray | None = None
        if result.face_landmarks:
            h, w = image_bgr.shape[:2]
            pts = result.face_landmarks[0]
            landmarks_xy = np.array(
                [[lm.x * w, lm.y * h] for lm in pts], dtype=np.float32
            )

        return LandmarkResult(blendshapes=blendshapes, landmarks_xy=landmarks_xy)
