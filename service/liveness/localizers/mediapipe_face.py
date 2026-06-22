"""MediaPipe BlazeFace localizer (Tasks API, short-range model).

Detects the best face in an image, computes a basic quality score,
and returns a `FaceCrop`. The crop itself is taken by `face_crop.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from service.liveness.domain.face import BoundingBox, FaceCrop, FaceQuality, Point
from service.liveness.domain.verdict import Reason
from service.liveness.exceptions import (
    FaceNotDetected,
    LocalizerError,
    ModelLoadError,
    WeightsHashMismatch,
    WeightsNotFound,
)
from service.liveness.face_crop import crop_with_padding
from service.liveness.infrastructure.loader import resolve as resolve_weight
from service.liveness.infrastructure.logging import get_logger


_log = get_logger(__name__)


# Empirical quality bounds (defaults; deployment values live in the
# thresholds module). Refined in Phase 3 against the internal dataset.
_BLUR_NORMALIZER = 200.0  # Laplacian variance value mapped to score 1.0
_BRIGHTNESS_MIN = 50.0
_BRIGHTNESS_MAX = 200.0


class MediaPipeFaceLocalizer:
    """Implements the `FaceLocalizer` Protocol with MediaPipe BlazeFace."""

    name = "mediapipe_blazeface"

    def __init__(
        self,
        *,
        padding: float = 0.20,
        min_face_size_px: int = 64,
        # 0.85 (vs MediaPipe default 0.5) drops weak detections (tiny
        # background, edge-partial, heavily occluded faces). Bona-fide
        # full faces score 0.90+; 0.5 was leaving false positives. If a
        # deployment over-rejects, override via the constructor.
        min_detection_confidence: float = 0.85,
    ) -> None:
        self._padding = padding
        self._min_face_size_px = min_face_size_px
        self._min_detection_confidence = min_detection_confidence
        self._detector: vision.FaceDetector | None = None

    def warmup(self) -> None:
        if self._detector is not None:
            return
        # Weights errors (missing/hash mismatch) propagate as-is — they
        # are deployment misconfiguration, not localizer faults. Anything
        # else is wrapped in ModelLoadError so the orchestrator degrades.
        try:
            model_path: Path = resolve_weight("blaze_face_short_range")
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self._min_detection_confidence,
            )
            self._detector = vision.FaceDetector.create_from_options(options)
        except (WeightsNotFound, WeightsHashMismatch):
            raise
        except Exception as exc:  # noqa: BLE001 — wrap and re-raise
            _log.exception("localizer.warmup_failed", extra={"localizer": self.name})
            raise ModelLoadError(
                f"MediaPipe localizer failed to warm up: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("localizer.ready", extra={"localizer": self.name})

    def locate(self, image: np.ndarray) -> FaceCrop | None:
        """Return the largest face that meets minimum size, else None.

        Pick the LARGEST face by bbox area (highest-confidence isn't
        always largest; for eKYC we want the subject closest to the
        camera). Detections smaller than `min_face_size_px` on either
        side are filtered before selection. Multi-face is just logged;
        we pick the largest and let the caller handle it.
        """
        if self._detector is None:
            self.warmup()
        assert self._detector is not None  # for type checker

        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._detector.detect(mp_image)
        except Exception as exc:  # noqa: BLE001 — wrap framework errors
            _log.exception("localizer.detect_failed", extra={"localizer": self.name})
            raise LocalizerError(
                f"MediaPipe detect() raised: {type(exc).__name__}: {exc}"
            ) from exc

        if not result.detections:
            _log.info("face.not_detected", extra={"localizer": self.name})
            return None

        # Collect viable bboxes (min size enforced) with their detection.
        viable: list[tuple[int, object, BoundingBox]] = []
        for det in result.detections:
            bb = self._extract_bbox(det, image.shape[:2])
            if bb is None:
                continue
            if min(bb.width, bb.height) < self._min_face_size_px:
                continue
            viable.append((bb.area, det, bb))

        if not viable:
            _log.info(
                "face.filtered_out",
                extra={
                    "localizer": self.name,
                    "n_raw": len(result.detections),
                    "min_face_size_px": self._min_face_size_px,
                },
            )
            return None

        # Pick the largest face. Multi-face count kept for telemetry.
        _, best, bbox = max(viable, key=lambda t: t[0])
        self._last_n_faces = len(viable)
        if len(viable) > 1:
            _log.info(
                "face.multiple_seen",
                extra={"localizer": self.name, "n_faces": len(viable)},
            )

        landmarks = tuple(
            Point(x=float(kp.x) * image.shape[1], y=float(kp.y) * image.shape[0])
            for kp in (getattr(best, "keypoints", None) or [])
        )

        # Completeness checks BEFORE padding, judging the raw detection.
        completeness_issues = _completeness_issues(
            bbox, landmarks, image.shape[:2]
        )

        _, clamped_bbox = crop_with_padding(image, bbox, padding=self._padding)
        crop = image[
            clamped_bbox.y : clamped_bbox.y_max,
            clamped_bbox.x : clamped_bbox.x_max,
        ]
        quality = _measure_quality(crop)

        if completeness_issues:
            _log.info(
                "face.incomplete",
                extra={"issues": list(completeness_issues)},
            )

        return FaceCrop(
            bbox=clamped_bbox,
            landmarks=landmarks,
            quality=quality,
            crop_path=None,
            completeness_issues=completeness_issues,
        )

    @staticmethod
    def _extract_bbox(detection, image_shape: tuple[int, int]) -> BoundingBox | None:
        bb = detection.bounding_box
        if bb is None:
            return None
        h, w = image_shape
        x = max(0, int(bb.origin_x))
        y = max(0, int(bb.origin_y))
        width = min(w - x, int(bb.width))
        height = min(h - y, int(bb.height))
        if width <= 0 or height <= 0:
            return None
        return BoundingBox(x=x, y=y, width=width, height=height)


def _completeness_issues(
    bbox: BoundingBox,
    landmarks: tuple[Point, ...],
    image_shape: tuple[int, int],
) -> tuple[str, ...]:
    """Geometric sanity checks; non-empty return = face unusable.

    1. Frame edge — unpadded bbox touches the image boundary, so the
       face is cropped by the frame.
    2. Landmarks inside bbox — a keypoint outside the raw bbox means the
       detector extrapolated onto background (partial faces).
    3. Geometry — eye-to-eye/bbox-width ratio in a plausible band and
       mouth below both eyes. Catches profile, upside-down, noisy faces.
    """
    issues: list[str] = []
    h, w = image_shape

    # 1. Frame edge: ~2% slack so a face barely touching the edge counts.
    edge_margin = max(2, min(bbox.width, bbox.height) // 50)
    if (
        bbox.x <= edge_margin
        or bbox.y <= edge_margin
        or bbox.x_max >= w - edge_margin
        or bbox.y_max >= h - edge_margin
    ):
        issues.append(Reason.FACE_PARTIAL_FRAME.value)

    # 2. Landmarks must lie inside the raw bbox, with slack since
    # BlazeFace keypoints are not pixel-perfect.
    if landmarks:
        slack_x = max(2, bbox.width // 40)
        slack_y = max(2, bbox.height // 40)
        for lm in landmarks:
            if (
                lm.x < bbox.x - slack_x
                or lm.x > bbox.x_max + slack_x
                or lm.y < bbox.y - slack_y
                or lm.y > bbox.y_max + slack_y
            ):
                issues.append(Reason.FACE_LANDMARK_OUT_OF_BBOX.value)
                break

    # 3. Geometric sanity. Need >=4 keypoints (right_eye, left_eye,
    # nose, mouth) per BlazeFace ordering.
    if len(landmarks) >= 4:
        right_eye, left_eye, nose, mouth = landmarks[0], landmarks[1], landmarks[2], landmarks[3]
        del nose  # unused, kept for clarity of indexing

        eye_distance = abs(left_eye.x - right_eye.x)
        eye_ratio = eye_distance / bbox.width if bbox.width > 0 else 0.0

        # Empirical band: 0.20 (oblique/partial) .. 0.55 (zoom);
        # bona-fide frontal lands around 0.30-0.45.
        if eye_ratio < 0.20 or eye_ratio > 0.55:
            issues.append(Reason.FACE_GEOMETRY_INVALID.value)
        else:
            # Mouth must sit below the LOWER eye (larger y); tolerance
            # covers tilted heads.
            tol = max(2, bbox.height // 40)
            if mouth.y < max(right_eye.y, left_eye.y) - tol:
                issues.append(Reason.FACE_GEOMETRY_INVALID.value)

    return tuple(issues)


def _measure_quality(crop: np.ndarray) -> FaceQuality:
    """Cheap, deterministic face-quality metrics."""
    if crop.size == 0:
        return FaceQuality(0.0, 0.0, 0.0, is_acceptable=False)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(1.0, lap_var / _BLUR_NORMALIZER)

    mean_lum = float(gray.mean())
    if mean_lum < _BRIGHTNESS_MIN or mean_lum > _BRIGHTNESS_MAX:
        brightness_score = 0.0
    else:
        # Triangle peak at the midpoint of the acceptable band.
        mid = (_BRIGHTNESS_MIN + _BRIGHTNESS_MAX) / 2.0
        span = (_BRIGHTNESS_MAX - _BRIGHTNESS_MIN) / 2.0
        brightness_score = max(0.0, 1.0 - abs(mean_lum - mid) / span)

    occlusion_score = 1.0  # Placeholder — real occlusion detection is future work.

    is_acceptable = blur_score >= 0.30 and brightness_score >= 0.20

    return FaceQuality(
        blur_score=blur_score,
        brightness_score=brightness_score,
        occlusion_score=occlusion_score,
        is_acceptable=is_acceptable,
    )
