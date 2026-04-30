"""Pipeline orchestrator: glue between the API and the four service modules.

Order matters:
    metadata → tampering → preprocessor → ocr
Tampering must run on RAW pixels (before preprocessor), or its forensic
signal is invalid. Metadata is byte-level and runs first because it's cheap.
"""
from __future__ import annotations

import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import UploadFile

from api.v1.schema.verify import BaseVerifyResponse
from service import policy, report_builder
from service.metadata import metadata
from service.ocr import ocr
from service.preprocessor import preprocessor
from service.tampering import tampering

DEST_PATH = "files"


def upload_file(file: UploadFile, id: str) -> Path:
    """Save uploaded file under files/<id>/. Returns the saved path."""
    dest_dir = Path(f"{DEST_PATH}/{id}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / file.filename
    with path.open(mode="wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f" > File saved: {path}")
    return path


def verify(files: list[UploadFile], id: str) -> BaseVerifyResponse:
    """Run the full pipeline on the uploaded files of a single document."""
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    paths: List[Path] = [upload_file(f, id) for f in files]

    artifacts_dir = Path(DEST_PATH) / id
    tampering_dir = artifacts_dir / "tampering"
    preprocessor_dir = artifacts_dir / "preprocessor"
    ocr_dir = artifacts_dir / "ocr"

    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    metadata_reports = metadata.extract(paths)
    timings["metadata"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    tampering_reports = tampering.analyze(paths, output_dir=tampering_dir)
    timings["tampering"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    processed_pages = preprocessor.process(paths, output_dir=preprocessor_dir)
    timings["preprocessor"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    ocr_paths = [
        preprocessor_dir / f"{Path(page.source).stem}_p{page.page_number}.png"
        for page in processed_pages
    ]
    ocr_results = ocr.process(ocr_paths, output_dir=ocr_dir)
    timings["ocr"] = int((time.perf_counter() - t0) * 1000)

    risk = policy.compute(
        metadata_reports, tampering_reports, processed_pages, ocr_results,
    )

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    timestamp = datetime.now(timezone.utc).isoformat()

    return report_builder.build(
        request_id=request_id,
        timestamp=timestamp,
        elapsed_ms=elapsed_ms,
        elapsed_ms_per_stage=timings,
        metadata_reports=metadata_reports,
        tampering_reports=tampering_reports,
        processed_pages=processed_pages,
        ocr_results=ocr_results,
        risk=risk,
        ocr_engine_name="paddleocr",
    )
