"""Moiré pattern detector — FFT-based, no weights, classical CV.

Detects the periodic screen-interference patterns left by replay attacks
(a face shown on a phone/monitor) that bona-fide skin texture lacks.

Method: grayscale the crop, take the 2D FFT magnitude spectrum, mask out
low frequencies (DC/illumination) and high frequencies (sensor noise),
then score = peak/mean energy in the mid-band — a clean face is smooth,
moiré creates sharp peaks. The ratio is mapped to [0, 1] via a sigmoid
centered on an empirical neutral point; recalibrate in Phase 3.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from service.liveness.domain.evidence import EngineOutput
from service.liveness.domain.face import FaceCrop
from service.liveness.domain.report import DetectorResult
from service.liveness.domain.verdict import Reason, Status, Verdict
from service.liveness.infrastructure.logging import get_logger


_log = get_logger(__name__)


# Mid-band window as a fraction of the spectrum half-extent. Tuned to
# capture screen moiré (~0.1-0.4 of nyquist) while rejecting
# illumination (low) and sensor noise (high).
_BAND_LOW_FRAC = 0.10
_BAND_HIGH_FRAC = 0.40

# Sigmoid mapping raw peak/mean ratio -> score in [0,1]. Calibrated
# 2026-05-21 (n_bona=43, n_spoof=32): bona median 11.97/p95 22.09,
# spoof median 15.19/p95 22.39. Centered between the medians; slope set
# so the discriminative range [8, 25] spans most of [0, 1] without
# saturation. See lab/reports/eval_2026-05-21_baseline.csv.
_SIGMOID_CENTER = 13.5
_SIGMOID_SLOPE = 0.30


class MoireDetector:
    """Implements the `Detector` Protocol.

    Stateless and weight-free, but routes its math through `_compute` so
    it stays unit-testable independently of the adapter wiring.
    """

    name = "moire"

    def warmup(self) -> None:
        # No-op — no model to load.
        return

    def analyze(
        self,
        crop_pixels: np.ndarray,
        face: FaceCrop,
        full_image: np.ndarray,
        image_path: str,
    ) -> DetectorResult:
        del face, full_image, image_path  # moiré uses crop_pixels only
        start = time.perf_counter()
        try:
            engine_output = _compute(crop_pixels)
            score = engine_output.score
            reasons: tuple[str, ...] = ()
            contribution = Verdict.ACCEPT
            if score >= 0.80:
                reasons = (Reason.MOIRE_PATTERN_DETECTED.value,)
                contribution = Verdict.HARD_REJECT
            elif score >= 0.50:
                reasons = (Reason.MOIRE_REVIEW.value,)
                contribution = Verdict.REVIEW

            return DetectorResult(
                name=self.name,
                status=Status.OK,
                score=score,
                confidence=float(engine_output.confidence),
                contribution=contribution,
                reasons=reasons,
                raw_metrics=dict(engine_output.raw_metrics),
                heatmap_path=None,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 — surface as FAILED
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


def _compute(crop: np.ndarray) -> EngineOutput:
    """Pure FFT pipeline — testable without the adapter wrapper."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Fixed size so band fractions map to a stable absolute freq range.
    target = 256
    gray = cv2.resize(gray, (target, target), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = min(cy, cx)

    band_low = _BAND_LOW_FRAC * max_radius
    band_high = _BAND_HIGH_FRAC * max_radius
    band_mask = (radius >= band_low) & (radius <= band_high)

    band_values = magnitude[band_mask]
    if band_values.size == 0 or band_values.mean() <= 1e-6:
        return EngineOutput(
            score=0.0,
            confidence=0.0,
            raw_metrics={"peak": 0.0, "mean": 0.0, "ratio": 0.0},
        )

    peak = float(band_values.max())
    mean = float(band_values.mean())
    ratio = peak / mean
    score = _sigmoid(ratio, center=_SIGMOID_CENTER, slope=_SIGMOID_SLOPE)

    return EngineOutput(
        score=float(score),
        confidence=1.0,
        raw_metrics={"peak": peak, "mean": mean, "ratio": ratio},
    )


def _sigmoid(x: float, *, center: float, slope: float) -> float:
    z = slope * (x - center)
    # Stable sigmoid.
    if z >= 0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)
