"""Translate the internal module outputs into the public BaseVerifyResponse."""
from __future__ import annotations

import json
from typing import List

from api.v1.schema.common import (
    ExecutionMetadataSchema,
    MetadataFileReportSchema,
    MetadataFlagSchema,
    MetadataModuleSchema,
    ModulesReportSchema,
    OCRModuleSchema,
    OCRPageSchema,
    PreprocessorModuleSchema,
    PreprocessorPageSchema,
    TamperingModuleSchema,
    TamperingPageSchema,
    Verdict,
)
from api.v1.schema.verify import BaseVerifyResponse
from service.metadata.analyzer import MetadataReport
from service.policy import RiskAggregate
from service.preprocessor.app.models import ProcessedPage
from service.tampering.detector import PageReport

PIPELINE_VERSION = "0.1.0"


def build(
    request_id: str,
    timestamp: str,
    elapsed_ms: int,
    elapsed_ms_per_stage: dict[str, int],
    metadata_reports: List[MetadataReport],
    tampering_reports: List[PageReport],
    processed_pages: List[ProcessedPage],
    ocr_results: list,
    risk: RiskAggregate,
    ocr_engine_name: str,
) -> BaseVerifyResponse:
    return BaseVerifyResponse(
        # Backwards-compatible flat fields
        tampering_score=risk.score,
        flags=risk.flags,
        confidence=risk.confidence,
        verdict=Verdict(risk.verdict),
        modules=ModulesReportSchema(
            metadata=_build_metadata(metadata_reports),
            tampering=_build_tampering(tampering_reports),
            preprocessor=_build_preprocessor(processed_pages),
            ocr=_build_ocr(ocr_results, ocr_engine_name),
        ),
        execution=ExecutionMetadataSchema(
            request_id=request_id,
            timestamp=timestamp,
            elapsed_ms=elapsed_ms,
            pipeline_version=PIPELINE_VERSION,
            elapsed_ms_per_stage=elapsed_ms_per_stage,
        ),
    )


def _build_metadata(reports: List[MetadataReport]) -> MetadataModuleSchema:
    files = [
        MetadataFileReportSchema(
            source=r.source,
            format=r.format,
            suspicion_score=r.suspicion_score,
            confidence=r.confidence,
            summary=r.summary,
            flags=[
                MetadataFlagSchema(
                    code=f.code,
                    severity=f.severity,
                    category=f.category,
                    evidence=f.evidence,
                )
                for f in r.flags
            ],
        )
        for r in reports
    ]
    aggregate = max((r.suspicion_score for r in reports), default=0.0)
    return MetadataModuleSchema(files=files, aggregate_suspicion=aggregate)


def _build_tampering(reports: List[PageReport]) -> TamperingModuleSchema:
    pages = [
        TamperingPageSchema(
            source=r.source,
            verdict=_str(r.verdict),
            verdict_score=r.verdict_score,
            confidence=_str(r.confidence),
            reasons=list(r.verdict_reasons),
            face_detected=bool(r.face.detected) if r.face else False,
            doctamper_score=r.doctamper.score_mean if r.doctamper and r.doctamper.ran else None,
            mvssnet_score=r.mvssnet.score if r.mvssnet and r.mvssnet.ran else None,
        )
        for r in reports
    ]
    rank = {"HARD_REJECT": 2, "REVIEW": 1, "ACCEPT": 0}
    worst = max(reports, key=lambda r: rank.get(_str(r.verdict), 0), default=None)
    worst_verdict = _str(worst.verdict) if worst is not None else "ACCEPT"
    worst_score = max((r.verdict_score for r in reports), default=0.0)
    return TamperingModuleSchema(
        pages=pages, worst_verdict=worst_verdict, worst_score=worst_score,
    )


def _build_preprocessor(pages: List[ProcessedPage]) -> PreprocessorModuleSchema:
    return PreprocessorModuleSchema(
        pages=[
            PreprocessorPageSchema(
                source=p.source,
                page_number=p.page_number,
                quality_score=round(p.quality.score, 3),
                quality_passed=p.quality.passed,
                rescued=p.quality.rescued,
                reason=p.quality.reason,
            )
            for p in pages
        ]
    )


def _build_ocr(results: list, engine_name: str) -> OCRModuleSchema:
    pages = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            pages.append(OCRPageSchema(page_number=i + 1))
            continue

        nested = r.get("result", {}) if isinstance(r.get("result"), dict) else {}

        document_type  = r.get("document_type")  or nested.get("document_type")
        verdict        = r.get("verdict")        or nested.get("verdict")
        confidence_avg = r.get("confidence_avg") or nested.get("confidence_avg", 0.0)
        fields         = r.get("fields")         or nested.get("fields")
        match_score    = r.get("match_score")
        flags          = r.get("flags")

        if fields:
            print(f"[OCR page {i + 1}] fields:\n{json.dumps(fields, indent=2, ensure_ascii=False)}")

        pages.append(OCRPageSchema(
            page_number    = i + 1,
            document_type  = document_type,
            verdict        = verdict,
            confidence_avg = confidence_avg,
            fields         = fields,
            match_score    = match_score,
            flags          = flags,
        ))
    return OCRModuleSchema(engine=engine_name, pages=pages)


def _str(value) -> str:
    """Coerce Enum or str to its string representation."""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
