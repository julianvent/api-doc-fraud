"""Calibrated thresholds — versioned, dataset-bound.

Rule (feedback memory): calibrated from p95 Real and p5 Spoof on a
balanced internal dataset; margin from `p95_real` must stay >= 0.015
(tampering lesson 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class DetectorThresholds:
    """Per-detector thresholds and their empirical provenance."""

    hard_reject_score: float
    review_score: float
    clean_override_score: float | None
    area_threshold: float | None
    p95_real: float
    p5_spoof: float
    safety_margin: float
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FaceQualityThresholds:
    blur_min: float
    brightness_min: float
    brightness_max: float
    face_min_size_px: int


@dataclass(frozen=True, slots=True)
class EnsembleThresholds:
    hard_reject_min_unique_signals: int
    review_min_unique_signals: int
    clean_override_enabled: bool
    clean_override_detector: str  # which detector confirms "clean"


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    dataset_version: str
    real_count: int
    spoof_count: int
    calibrated_at: date
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ThresholdsConfig:
    """Top-level versioned threshold bundle."""

    version: str
    calibration: CalibrationMetadata
    face_quality: FaceQualityThresholds
    ensemble: EnsembleThresholds
    detectors: dict[str, DetectorThresholds]


# Calibrated 2026-05-21 on the internal dataset (43 bona-fide + 32
# spoof_replay; 1 spoof failed face detection). Active ensemble: moire
# (FFT, soft only) + minifas (CNN, P(replay) as spoof score). Operating
# point: APCER 0 %, BPCER 2.4 %, ACER 1.22 %. Full comparison in
# lab/reports/2026-05-21_phase2c_triple_minifas.md.
INTERNAL_V1 = ThresholdsConfig(
    version="0.3.0-internal-moire-minifas-v1",
    calibration=CalibrationMetadata(
        dataset_version="internal-2026-05-21",
        real_count=43,
        spoof_count=32,
        calibrated_at=date(2026, 5, 21),
        notes="Active ensemble: moire (FFT, soft only) + minifas (CNN, P(replay)).",
    ),
    face_quality=FaceQualityThresholds(
        blur_min=0.30,
        brightness_min=0.20,
        brightness_max=0.95,
        face_min_size_px=64,
    ),
    ensemble=EnsembleThresholds(
        hard_reject_min_unique_signals=2,
        review_min_unique_signals=1,
        clean_override_enabled=True,
        clean_override_detector="minifas",
    ),
    detectors={
        "moire": DetectorThresholds(
            # Moire is the WEAKER detector: 9/41 bona_fide score above
            # 0.64 on Moire alone (all caught by MiniFAS), so a Tier 1
            # veto here would mean 9 false positives. Hard threshold set
            # above the empirical max so Moire never vetoes alone; it
            # contributes only as a soft signal via review_score.
            hard_reject_score=1.10,
            review_score=0.477,
            clean_override_score=None,
            area_threshold=None,
            p95_real=0.929,
            p5_spoof=0.285,
            safety_margin=0.015,
            notes=(
                "Soft-signal only. Review at Youden/ACER point 0.477. "
                "Hard reject disabled (1.10) because Moire alone is too "
                "noisy to veto unilaterally; ensemble promotion takes over."
            ),
        ),
        # Score = P(replay) from MiniFASNetV2 softmax (NOT 1-P(live),
        # which saturates at ~1.0 for all inputs since the model is biased
        # toward `print` on our WhatsApp-compressed bona_fides). P(replay)
        # separates with AUC=0.997; see minifas.py for the OOD discussion.
        # Distributions: bona_fide median=0.00, max=0.51; spoof min=0.25,
        # median=1.00. Solo cut (hard=0.8, review=0.5): APCER=6.9% BPCER=0%.
        "minifas": DetectorThresholds(
            hard_reject_score=0.80,
            review_score=0.50,
            clean_override_score=0.10,
            area_threshold=None,
            p95_real=0.000,
            p5_spoof=0.551,
            safety_margin=0.015,
            notes=(
                "P(replay) score, internal-2026-05-21. APCER=6.9% BPCER=0% solo. "
                "CAVEAT: print attacks not in scope — switch to "
                "max(p_print, p_replay) if dataset grows to include prints."
            ),
        ),
    },
)


CURRENT_THRESHOLDS = INTERNAL_V1
