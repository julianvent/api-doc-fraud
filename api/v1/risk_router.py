"""Router for enterprise risk threshold configuration.

GET  /v1/risk/thresholds  — return the active threshold configuration.
PUT  /v1/risk/thresholds  — replace the active threshold configuration.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.v1.schema.risk_thresholds import RiskThresholdsRequest, RiskThresholdsResponse
import service.risk_thresholds_store as _store

router = APIRouter(prefix="/v1/risk", tags=["risk"])


@router.get("/thresholds", response_model=RiskThresholdsResponse)
async def get_thresholds() -> RiskThresholdsResponse:
    t = _store.get()
    return RiskThresholdsResponse(
        approve_max=t.approve_max,
        review_max=t.review_max,
        edd_max=t.edd_max,
    )


@router.put("/thresholds", response_model=RiskThresholdsResponse)
async def update_thresholds(body: RiskThresholdsRequest) -> RiskThresholdsResponse:
    updated = _store.update(
        approve_max=body.approve_max,
        review_max=body.review_max,
        edd_max=body.edd_max,
    )
    return RiskThresholdsResponse(
        approve_max=updated.approve_max,
        review_max=updated.review_max,
        edd_max=updated.edd_max,
    )
