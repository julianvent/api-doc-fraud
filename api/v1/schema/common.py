"""Reusable enums and sub-schemas for the public API response."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


# ── metadata ──────────────────────────────────────────────────────────────────

class MetadataFlagSchema(BaseModel):
    code: str
    severity: str
    category: str
    evidence: str


class MetadataFileReportSchema(BaseModel):
    source: str
    format: str
    suspicion_score: float
    confidence: str
    summary: str
    flags: List[MetadataFlagSchema]


class MetadataModuleSchema(BaseModel):
    files: List[MetadataFileReportSchema]
    aggregate_suspicion: float


# ── tampering ─────────────────────────────────────────────────────────────────

class TamperingPageSchema(BaseModel):
    source: str
    risk_label: str
    fraud_score: float
    reliability: str
    reasons: List[str]
    face_detected: bool
    doctamper_score: Optional[float] = None
    trufor_score: Optional[float] = None
    face_trufor_score: Optional[float] = None


class TamperingModuleSchema(BaseModel):
    pages: List[TamperingPageSchema]
    worst_risk_label: str
    worst_fraud_score: float


# ── preprocessor ──────────────────────────────────────────────────────────────

class PreprocessorPageSchema(BaseModel):
    source: str
    page_number: int
    quality_score: float
    quality_passed: bool
    rescued: bool
    reason: str


class PreprocessorModuleSchema(BaseModel):
    pages: List[PreprocessorPageSchema]


# ── ocr ───────────────────────────────────────────────────────────────────────

class OCRPageSchema(BaseModel):
    page_number: int
    document_type: Optional[str]
    verdict: Optional[str]
    confidence_avg: float


class OCRModuleSchema(BaseModel):
    engine: str
    pages: List[OCRPageSchema]


# ── modules wrapper ───────────────────────────────────────────────────────────

class ModulesReportSchema(BaseModel):
    metadata: MetadataModuleSchema
    tampering: TamperingModuleSchema
    preprocessor: PreprocessorModuleSchema
    ocr: OCRModuleSchema


class ExecutionMetadataSchema(BaseModel):
    request_id: str
    timestamp: str
    elapsed_ms: int
    pipeline_version: str
    elapsed_ms_per_stage: dict[str, int]
