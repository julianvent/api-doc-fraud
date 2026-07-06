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
    ConsistencyVerification,
)
from api.v1.schema.verify import BaseVerifyResponse
from service.metadata.analyzer import MetadataReport
from service.policy import RiskAggregate
from service.preprocessor.app.models import ProcessedPage
from service.tampering.detector import PageReport, RiskLabel

_RISK_RANK = {
    RiskLabel.LIKELY_MANIPULATED: 2,
    RiskLabel.SUSPICIOUS: 1,
    RiskLabel.LEGITIMATE: 0,
}

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
        risk_score=risk.score,
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
    pages = [
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
    return MetadataModuleSchema(pages=pages, aggregate_suspicion=aggregate)


def _build_tampering(reports: List[PageReport]) -> TamperingModuleSchema:
    from pathlib import Path as _Path

    pages = [
        TamperingPageSchema(
            source=r.source,
            risk_label=r.risk_label.value,
            fraud_score=r.fraud_score,
            reliability=r.reliability.value,
            reasons=[f"[{f.severity.value}] {f.message}" for f in r.findings],
            face_detected=bool(r.face.detected) if r.face else False,
            doctamper_score=(
                r.doctamper.score_mean if r.doctamper and r.doctamper.ran else None
            ),
            trufor_score=r.trufor.score if r.trufor and r.trufor.ran else None,
            face_trufor_score=(
                r.face_trufor.score if r.face_trufor and r.face_trufor.ran else None
            ),
            overlay_filename=(
                _Path(r.artifacts.doctamper_overlay).name
                if r.artifacts and r.artifacts.doctamper_overlay
                else None
            ),
        )
        for r in reports
    ]
    worst = max(reports, key=lambda r: _RISK_RANK.get(r.risk_label, 0), default=None)
    worst_risk_label = (
        worst.risk_label.value if worst is not None else RiskLabel.LEGITIMATE.value
    )
    worst_fraud_score = max((r.fraud_score for r in reports), default=0.0)
    return TamperingModuleSchema(
        pages=pages,
        worst_risk_label=worst_risk_label,
        worst_fraud_score=worst_fraud_score,
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
    identity_mismatches = []
    mrz_mismatches = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            pages.append(OCRPageSchema(page_number=i + 1))
            continue

        nested = r.get("result", {}) if isinstance(r.get("result"), dict) else {}

        document_type = r.get("document_type") or nested.get("document_type")
        ocr_confidence = r.get("ocr_confidence") or nested.get("ocr_confidence", 0.0)
        fields = r.get("fields") or nested.get("fields")
        extras = r.get("extras") or nested.get("extras")
        template_match_confidence = r.get("template_match_confidence")
        flags = r.get("flags")
        
        
        if r.get("identity_mismatches"):
            identity_mismatches.extend(r.get("identity_mismatches"))
        if r.get("mrz_mismatches"):
            mrz_mismatches.extend(r.get("mrz_mismatches"))

        if fields:
            print(
                f"[OCR page {i + 1}] fields:\n{json.dumps(fields, indent=2, ensure_ascii=False)}"
            )

        pages.append(
            OCRPageSchema(
                page_number=i + 1,
                document_type=document_type,
                ocr_confidence=ocr_confidence,
                fields=fields,
                extras=extras,
                template_match_confidence=template_match_confidence,
                flags=flags,
            )
        )
        
    return OCRModuleSchema(
        engine=engine_name,
        pages=pages,
        consistency_verification=ConsistencyVerification(
            consistency= not (identity_mismatches or mrz_mismatches),
            identity_inconsistencies=identity_mismatches,
            mrz_inconsistencies=mrz_mismatches,
        ),
    )
