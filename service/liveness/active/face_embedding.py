"""Face embedding for identity consistency across an active session.

Wraps insightface `w600k_mbf` (MobileFaceNet, ArcFace) via onnxruntime
into a 512-d L2-normalized embedding; cosine is high (~0.7) for the same
person, low (<0.3) for different people.

WHY: confirm the face that starts the check is the one that finishes it,
so an attacker can't center a real face then swap in another person/video
for the motion. A geometric "fingerprint" was tried first and did not
discriminate identity at all; the learned embedding splits cleanly
(same-person cosine median 0.70 vs different-person max 0.30).

ALIGNMENT: ArcFace needs a 5-point similarity transform to a canonical
112x112 template — a plain bbox crop degrades it badly. We align using
FaceLandmarker's 5 key points.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from service.liveness.exceptions import (
    ModelLoadError,
    WeightsHashMismatch,
    WeightsNotFound,
)
from service.liveness.infrastructure.loader import resolve as resolve_weight
from service.liveness.infrastructure.logging import get_logger


_log = get_logger(__name__)


# ArcFace canonical 5-point template for a 112x112 crop, in
# (image-left eye, image-right eye, nose, image-left mouth,
# image-right mouth) order.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# MediaPipe FaceLandmarker indices for the 5 alignment points.
_RIGHT_EYE_CORNERS = (33, 133)
_LEFT_EYE_CORNERS = (362, 263)
_NOSE_TIP = 1
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291

# Same- vs different-person cosine threshold. Calibrated on the internal
# dataset (7 identities: intra-min 0.44, inter-max 0.30): 0.35 sits in
# the clean gap and matches the ArcFace verification range (0.28-0.36).
IDENTITY_MATCH_THRESHOLD = 0.35

_INPUT_SIZE = 112


class FaceEmbedder:
    """Thin onnxruntime wrapper exposing `embed(image, landmarks_xy)`."""

    name = "insightface_w600k_mbf"

    def __init__(self) -> None:
        self._session = None
        self._input_name: str | None = None

    def warmup(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort

            model_path: Path = resolve_weight("face_recognition")
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except (WeightsNotFound, WeightsHashMismatch):
            raise
        except Exception as exc:  # noqa: BLE001
            _log.exception("face_embedder.warmup_failed", extra={"model": self.name})
            raise ModelLoadError(
                f"FaceEmbedder failed to warm up: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("face_embedder.ready", extra={"model": self.name})

    def embed(
        self, image_bgr: np.ndarray, landmarks_xy: np.ndarray | None
    ) -> np.ndarray | None:
        """Return an L2-normalized 512-d embedding, or None if unusable.

        `landmarks_xy` is the (N, 2) array from FaceLandmarker; the 5
        alignment points are read from it.
        """
        if self._session is None:
            self.warmup()
        assert self._session is not None

        src = _five_points(landmarks_xy)
        if src is None:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_TEMPLATE)
        if matrix is None:
            return None
        aligned = cv2.warpAffine(
            image_bgr, matrix, (_INPUT_SIZE, _INPUT_SIZE), borderValue=0
        )
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        try:
            out = self._session.run(None, {self._input_name: blob})[0][0]
        except Exception:  # noqa: BLE001
            _log.exception("face_embedder.inference_failed")
            return None
        norm = float(np.linalg.norm(out))
        if norm < 1e-9:
            return None
        return (out / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalized embeddings (a dot product)."""
    return float(np.dot(a, b))


def is_same_identity(
    a: np.ndarray | None,
    b: np.ndarray | None,
    threshold: float = IDENTITY_MATCH_THRESHOLD,
) -> bool:
    """True if both embeddings exist and their cosine >= threshold.

    Also True when either is missing: "cannot evaluate" must not by
    itself fail the session; the caller (session) just skips the check
    on frames without an embedding.
    """
    if a is None or b is None:
        return True
    return cosine_similarity(a, b) >= threshold


def _five_points(landmarks_xy: np.ndarray | None) -> np.ndarray | None:
    """Extract the 5 ArcFace alignment points from FaceLandmarker output.

    Eyes and mouth corners are assigned by x-coordinate so the result
    matches the template regardless of MediaPipe's subject-vs-image
    handedness convention.
    """
    needed = max(
        max(_RIGHT_EYE_CORNERS), max(_LEFT_EYE_CORNERS),
        _NOSE_TIP, _MOUTH_LEFT, _MOUTH_RIGHT,
    )
    if landmarks_xy is None or len(landmarks_xy) <= needed:
        return None

    def _center(idxs):
        return np.mean([landmarks_xy[i] for i in idxs], axis=0)

    right_eye = _center(_RIGHT_EYE_CORNERS)
    left_eye = _center(_LEFT_EYE_CORNERS)
    nose = landmarks_xy[_NOSE_TIP]
    mouth_l = landmarks_xy[_MOUTH_LEFT]
    mouth_r = landmarks_xy[_MOUTH_RIGHT]

    eyes = np.array([right_eye, left_eye], dtype=np.float32)
    eyes = eyes[np.argsort(eyes[:, 0])]          # image-left eye first
    mouths = np.array([mouth_l, mouth_r], dtype=np.float32)
    mouths = mouths[np.argsort(mouths[:, 0])]    # image-left mouth corner first

    return np.array(
        [eyes[0], eyes[1], nose, mouths[0], mouths[1]], dtype=np.float32
    )
