"""Request and response schemas for PUT /v1/risk/thresholds."""
from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class RiskThresholdsRequest(BaseModel):
    approve_max: float
    review_max:  float
    edd_max:     float

    @field_validator("approve_max", "review_max", "edd_max")
    @classmethod
    def in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("threshold must be strictly between 0.0 and 1.0")
        return round(v, 4)

    @model_validator(mode="after")
    def ascending_order(self) -> RiskThresholdsRequest:
        if not (self.approve_max < self.review_max < self.edd_max):
            raise ValueError(
                "thresholds must satisfy approve_max < review_max < edd_max"
            )
        return self


class RiskThresholdsResponse(BaseModel):
    approve_max: float
    review_max:  float
    edd_max:     float
