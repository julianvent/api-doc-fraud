"""MediaPipe selfie-multiclass parser wrapper.

Returns a per-pixel category mask using the `selfie_multiclass_256x256`
model. The model emits 6 classes:

  0 : background
  1 : hair
  2 : body-skin (neck, shoulders)
  3 : face-skin   <- the only class that is acceptable inside the face oval
  4 : clothes
  5 : others (glasses, hats, headphones, microphones, hand-held objects)

`liveness/active/occlusion.py` consumes the mask + face-landmark polygon
to count non-face-skin pixels inside the face oval — that fraction is the
direct, single-source-of-truth occlusion signal.

This replaces the prior heuristic stack (bridge variance, mouth
saturation, eye Laplacian, color-anchor), each an indirect proxy with its
own failure mode, with one direct per-pixel semantic label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
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


class FaceParseClass(IntEnum):
    """Class indices emitted by selfie_multiclass_256x256.

    The order matches MediaPipe's model documentation. Do NOT renumber.
    """

    BACKGROUND = 0
    HAIR = 1
    BODY_SKIN = 2
    FACE_SKIN = 3
    CLOTHES = 4
    OTHERS = 5


@dataclass(frozen=True, slots=True)
class FaceParseResult:
    """Output of one parse() call.

    `category_mask` is a 2D uint8 array of shape (H, W) where each
    element is a `FaceParseClass` value. Same resolution as the input
    image. `None` when the model failed to produce a mask (rare).
    """

    category_mask: np.ndarray | None


_EMPTY = FaceParseResult(category_mask=None)


class MediaPipeFaceParser:
    """Thin wrapper exposing `parse(image_bgr)` only."""

    name = "mediapipe_selfie_multiclass"

    def __init__(self) -> None:
        self._segmenter: vision.ImageSegmenter | None = None

    def warmup(self) -> None:
        if self._segmenter is not None:
            return
        try:
            model_path: Path = resolve_weight("selfie_multiclass")
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=False,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
        except (WeightsNotFound, WeightsHashMismatch):
            raise
        except Exception as exc:  # noqa: BLE001 — wrap framework errors
            _log.exception("face_parser.warmup_failed", extra={"model": self.name})
            raise ModelLoadError(
                f"FaceParser failed to warm up: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("face_parser.ready", extra={"model": self.name})

    def parse(self, image_bgr: np.ndarray) -> FaceParseResult:
        """Return a per-pixel class mask at the input image resolution."""
        if self._segmenter is None:
            self.warmup()
        assert self._segmenter is not None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._segmenter.segment(mp_image)
        except Exception as exc:  # noqa: BLE001
            _log.exception("face_parser.segment_failed")
            return _EMPTY

        if result.category_mask is None:
            return _EMPTY
        mask = np.copy(result.category_mask.numpy_view())
        return FaceParseResult(category_mask=mask)
