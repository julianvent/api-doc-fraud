"""Thresholds calibrated against the 42-doc dataset (25 Real / 17 Fake).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class Thresholds:
    # ─── DocTamper raw output shaping ────────────────────────────────────
    # Review entry. Above Real p80 so security textures don't fire alone.
    pass_threshold: float = 0.45
    # Per-pixel binarization for extracting suspicious regions.
    binary_threshold: float = 0.70
    # Drop connected components smaller than this fraction of image area.
    min_region_area_fraction: float = 0.005

    # ─── Decision thresholds: text tampering (DocTamper) ────────────────
    # Tier-1 reject on mean. Above Real max (0.479); catches DT>=0.50 fakes.
    text_global_reject_score: float = 0.50
    # A region needs this local score to be "strong".
    text_reject_score: float = 0.90
    # Strong localized regions required for Tier-1 reject.
    text_reject_region_count: int = 2
    # Above this fraction the region is "large" — usually scan artifact.
    max_region_area_fraction_for_reject: float = 0.05
    # Outside-face score that routes to review.
    text_review_score: float = 0.50

    # ─── Decision: ensemble-level aggregation ───────────────────────────
    # Independent MEDIUM+ findings that collectively promote to reject.
    ensemble_reject_signal_count: int = 4

    # ─── Decision thresholds: image manipulation (TruFor) ───────────────
    # Tier-1 reject on score. ~p95 of Real — security textures stay below.
    trufor_reject_score: float = 0.88
    # Review band start.
    trufor_review_score: float = 0.62
    # Largest contiguous suspicious area (heatmap >= 0.5) for Tier-1.
    trufor_reject_min_area_fraction: float = 0.06
    # Review trigger on area alone.
    trufor_review_min_area_fraction: float = 0.02

    # ─── Decision thresholds: TruFor on the face crop ───────────────────
    # Tier-1 face manipulation. Above Real max (0.285); catches visafake*.
    face_trufor_reject_score: float = 0.30
    face_trufor_reject_min_area_fraction: float = 0.15

    # ─── Decision: counter-signal (clean face overrides soft signals) ───
    # FaceTF "clean" cutoff. Only Real and a few clean Fakes sit here.
    face_trufor_clean_score: float = 0.12
    face_trufor_clean_area_fraction: float = 0.01

    def as_dict(self) -> Dict[str, float]:
        """Return a plain dict for snapshotting into metadata."""
        return {k: float(v) for k, v in asdict(self).items()}


DEFAULT = Thresholds()
