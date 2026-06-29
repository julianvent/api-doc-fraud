"""Public request and response schemas for POST /v1/verify/."""

from __future__ import annotations

from typing import List

from fastapi import UploadFile
from pydantic import BaseModel

from api.v1.schema.common import (
    ExecutionMetadataSchema,
    ModulesReportSchema,
    Verdict,
)


class BaseVerifyRequest(BaseModel):
    document_images: list[UploadFile]
    id: str
    document_type: str | None = None
    full_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None


class BaseVerifyResponse(BaseModel):
    # Backwards-compatible flat fields. Existing consumers keep working.
    tampering_score: float
    flags: List[str]
    confidence: float

    # Expanded fields for the demo.
    verdict: Verdict
    modules: ModulesReportSchema
    execution: ExecutionMetadataSchema


class Identity(BaseModel):
    full_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
