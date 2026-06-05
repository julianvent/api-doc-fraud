"""MiniFASNetV2 anti-spoof detector.

Wraps the vendored MiniFASNetV2 and the 2.7_80x80_MiniFASNetV2.pth
checkpoint from Minivision's Silent-Face-Anti-Spoofing repo.

Preprocessing matches upstream `CropImage(scale=2.7)`: expand the bbox
2.7x around the face center (clamped), then resize to 80x80. The net is
trained on that exact context window; a tight crop is out-of-distribution
and the score collapses to noise.

3-way softmax: 0=live, 1=2D print attack, 2=3D/replay attack. Reported
spoof score is P(replay).

WHY P(replay), NOT 1 - P(live): empirically (2026-05-21, 41 bona_fide +
29 spoof_replay) the model is print-biased, scoring WhatsApp-compressed
bona-fide selfies as p_print ~ 1.0; so P(live) ~ 0 everywhere (AUC ~
0.46) while P(replay) is orthogonal (AUC = 0.997). CAVEAT: works only
because our spoofs are replay-only. A real print would score p_print~1,
p_replay~0 and look "clean" — then fine-tune or switch the score to
max(p_print, p_replay) and recalibrate.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from service.liveness.domain.face import FaceCrop
from service.liveness.domain.report import DetectorResult
from service.liveness.domain.verdict import Reason, Status, Verdict
from service.liveness.exceptions import (
    InferenceError,
    ModelLoadError,
    WeightsHashMismatch,
    WeightsNotFound,
)
from service.liveness.infrastructure.loader import resolve as resolve_weight
from service.liveness.infrastructure.logging import get_logger
from service.liveness.models.minifasnet import MiniFASNetV2, get_kernel


_log = get_logger(__name__)


_INPUT_SIZE = 80
# Upstream trained on raw-bbox expanded 2.7x. Our face.bbox already
# carries 20% localizer padding, so using 2.7 here would be ~3.24x
# (out-of-distribution). 2.7 / 1.2 ~ 2.25 restores the expected crop.
_SCALE = 2.25
_NUM_CLASSES = 3


def _expanded_face_crop(
    image: np.ndarray, face: FaceCrop, scale: float = _SCALE
) -> np.ndarray:
    """Replicate upstream `CropImage.crop()` with the given scale.

    The FaceCrop bbox is already 20%-padded; applying the full scale to
    it is close enough to the training recipe for in-distribution
    inference. If accuracy is poor, thread the raw unpadded bbox through.
    """
    src_h, src_w = image.shape[:2]
    bb = face.bbox
    box_w, box_h = bb.width, bb.height
    # Bound the effective scale by image extent (upstream does this too).
    eff_scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * eff_scale
    new_h = box_h * eff_scale
    cx = bb.x + box_w / 2.0
    cy = bb.y + box_h / 2.0

    left = cx - new_w / 2.0
    top = cy - new_h / 2.0
    right = cx + new_w / 2.0
    bottom = cy + new_h / 2.0

    # Slide back into the image if any corner overshoots.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1

    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = int(round(right)), int(round(bottom))
    patch = image[y0 : y1 + 1, x0 : x1 + 1]
    return cv2.resize(patch, (_INPUT_SIZE, _INPUT_SIZE))


class MiniFASDetector:
    """Implements the `Detector` Protocol."""

    name = "minifas"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cpu")

    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            weights = resolve_weight("minifas_v2_2_7_80x80")
            kernel = get_kernel(_INPUT_SIZE, _INPUT_SIZE)
            model = MiniFASNetV2(
                conv6_kernel=kernel, num_classes=_NUM_CLASSES
            ).to(self._device)
            state_dict = torch.load(
                weights, map_location=self._device, weights_only=True
            )
            # Upstream checkpoint may carry the `module.` prefix from DataParallel training.
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        except (WeightsNotFound, WeightsHashMismatch):
            raise
        except Exception as exc:  # noqa: BLE001 — wrap and re-raise
            _log.exception("detector.warmup_failed", extra={"detector": self.name})
            raise ModelLoadError(
                f"MiniFASNetV2 failed to warm up: {type(exc).__name__}: {exc}"
            ) from exc
        _log.info("detector.ready", extra={"detector": self.name})

    def analyze(
        self,
        crop_pixels: np.ndarray,
        face: FaceCrop,
        full_image: np.ndarray,
        image_path: str,
    ) -> DetectorResult:
        del crop_pixels, image_path  # minifas re-crops at scale=2.7 from full_image
        start = time.perf_counter()
        try:
            if self._model is None:
                self.warmup()
            assert self._model is not None

            patch = _expanded_face_crop(full_image, face, scale=_SCALE)
            # IMPORTANT: Minivision's `to_tensor` does NOT divide by 255
            # (src/data_io/functional.py); the model is trained on
            # [0, 255]. Feeding [0, 1] collapses the softmax to one class.
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
            tensor = tensor.to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            p_live = float(probs[0])
            p_print = float(probs[1])
            p_replay = float(probs[2])
            # See module docstring for the rationale behind P(replay)
            # over (1 - P(live)).
            spoof_score = p_replay

            return DetectorResult(
                name=self.name,
                status=Status.OK,
                score=spoof_score,
                confidence=1.0,
                contribution=Verdict.ACCEPT,
                reasons=(),
                raw_metrics={
                    "p_live": p_live,
                    "p_print": p_print,
                    "p_replay": p_replay,
                    "spoof_score": spoof_score,
                },
                heatmap_path=None,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except WeightsNotFound:
            raise
        except Exception as exc:  # noqa: BLE001
            _log.exception(
                "detector.inference_failed",
                extra={"detector": self.name, "exc_type": type(exc).__name__},
            )
            return DetectorResult(
                name=self.name,
                status=Status.FAILED,
                score=0.0,
                confidence=0.0,
                contribution=Verdict.REVIEW,
                reasons=(),
                raw_metrics={},
                heatmap_path=None,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
