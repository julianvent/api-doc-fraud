"""In-memory store for enterprise risk threshold configuration.

Thresholds define score band boundaries for the 4-tier verdict system:
  score <= approve_max           -> ACCEPT
  approve_max < score <= review_max -> REVIEW
  review_max < score <= edd_max  -> EDD
  score > edd_max                -> REJECT (score-based path)

Hard rules in policy.py can still short-circuit directly to REJECT or REVIEW
regardless of these thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_APPROVE_MAX = 0.30   # scores 0–30 → ACCEPT
_DEFAULT_REVIEW_MAX  = 0.60   # scores 31–60 → REVIEW
_DEFAULT_EDD_MAX     = 0.80   # scores 61–80 → EDD; above → REJECT


@dataclass(frozen=True)
class RiskThresholds:
    approve_max: float = _DEFAULT_APPROVE_MAX
    review_max:  float = _DEFAULT_REVIEW_MAX
    edd_max:     float = _DEFAULT_EDD_MAX


_store: RiskThresholds = RiskThresholds()


def get() -> RiskThresholds:
    return _store


def update(approve_max: float, review_max: float, edd_max: float) -> RiskThresholds:
    global _store
    _store = RiskThresholds(
        approve_max=approve_max,
        review_max=review_max,
        edd_max=edd_max,
    )
    return _store


def reset() -> RiskThresholds:
    global _store
    _store = RiskThresholds()
    return _store
