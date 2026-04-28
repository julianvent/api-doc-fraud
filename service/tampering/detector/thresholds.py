"""Configurable thresholds for tampering detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class Thresholds:
    # ─── DocTamper raw output shaping ────────────────────────────────────
    # Mean tampered probability above which the page is no longer auto-ACCEPT
    # by the text detector alone. 
    pass_threshold: float = 0.20
    # Per-pixel binarization threshold used to extract suspicious regions.
    # Raised to 0.70 so only high-confidence pixels form regions; lower values
    # inflate region size on scanned / re-compressed documents where the model
    # emits a noisy baseline across text lines.
    binary_threshold: float = 0.70
    # Drop connected components smaller than this fraction of image area.
    # Lowered so genuinely small, surgical edits (a single field, a stamp) are
    # still captured.
    min_region_area_fraction: float = 0.005

    # ─── Decision thresholds: text tampering (DocTamper) ────────────────
    # DocTamper mean over the whole page above this level promotes directly
    # to HARD_REJECT. Authentic documents sit in ~0.12-0.19 and even heavily
    # compressed scans rarely exceed ~0.25 — a mean this high indicates the
    # model sees most of the page as tampered, which is the signature of
    # fabricated or AI-generated content.
    text_global_reject_score: float = 0.40
    # A region must reach this local score to be considered "strong".
    text_reject_score: float = 0.85
    # Minimum count of strong *and* localized regions to promote the page to
    # HARD_REJECT. Requiring two or more strong localized hits keeps recall
    # while rejecting single-region scan artifacts.
    text_reject_region_count: int = 2
    # Regions larger than this fraction of the page do NOT count toward
    # HARD_REJECT even when their score crosses `text_reject_score`. Real
    # tampering is surgical (<5% of the page); large hot regions are
    # overwhelmingly scan / compression artifacts. Such regions still flow
    # through the REVIEW tier so a human can confirm.
    max_region_area_fraction_for_reject: float = 0.05
    # Minimum score_outside_face that pushes the page to REVIEW even when
    # HARD_REJECT is not triggered.
    text_review_score: float = 0.30

    # ─── Decision: ensemble-level aggregation ───────────────────────────
    # Count of independent REVIEW-level reasons that, combined, promote the
    # page to HARD_REJECT. Captures fabricated documents where no single
    # detector crosses its own reject threshold but multiple fire together.
    ensemble_reject_signal_count: int = 4

    # ─── Decision thresholds: photo splicing (MVSS-Net) ─────────────────
    # Score is now `mean of top-1% pixels` (engine.MVSSNetEngine), not max.
    # Thresholds below are calibrated for that more robust metric.
    photo_reject_score: float = 0.70
    photo_review_score: float = 0.40
    # HARD_REJECT additionally requires the largest contiguous high-probability
    # region to cover at least this fraction of the face crop. Real spliced
    # photos produce one sizable cluster; scanner noise produces many tiny
    # scattered components. Requiring mass in a single region rejects the
    # noise case while keeping the tampering case.
    photo_reject_min_area_fraction: float = 0.05
    # Below this area any score still routes to REVIEW (not auto-ACCEPT),
    # so a human can confirm ambiguous cases.
    photo_review_min_area_fraction: float = 0.02

    def as_dict(self) -> Dict[str, float]:
        """Return a plain dict suitable for snapshotting into metadata."""
        return {k: float(v) for k, v in asdict(self).items()}


DEFAULT = Thresholds()
