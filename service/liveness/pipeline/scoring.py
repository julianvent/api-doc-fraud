"""Score aggregation — pure functions over DetectorResult collections."""

from __future__ import annotations

from collections.abc import Iterable

from service.liveness.domain.report import DetectorResult
from service.liveness.domain.verdict import Status


def aggregate_score(results: Iterable[DetectorResult]) -> float:
    """Aggregate per-detector scores into a single module-level score.

    V1 strategy: max of OK detector scores — simple, interpretable,
    survives one misbehaving detector. Replace with a calibrated weighted
    mean once the dataset is large enough (Phase 5+). Returns 0.0 when no
    detector ran (caller treats that as missing signal, usually REVIEW).
    """
    ok_scores = [r.score for r in results if r.status == Status.OK]
    if not ok_scores:
        return 0.0
    return max(ok_scores)
