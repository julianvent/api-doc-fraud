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
    verdict: str
    verdict_score: float
    confidence: str
    reasons: List[str]
    face_detected: bool
    doctamper_score: Optional[float] = None
    mvssnet_score: Optional[float] = None


class TamperingModuleSchema(BaseModel):
    pages: List[TamperingPageSchema]
    worst_verdict: str
    worst_score: float


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
    english_text: str
    hindi_text: str
    english_words_count: int
    hindi_words_count: int
    low_confidence_count: int


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
