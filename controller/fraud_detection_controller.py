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
from service.logging_config import get_logger
from service.metadata import metadata
from service.ocr import ocr
from service.preprocessor import preprocessor
from service.preprocessor.app.io.writer import save_image
from service.tampering import tampering
from model.document_template import DocumentTemplate  # noqa: F401  (kept for legacy callers)


log = get_logger(__name__)

FILES_PATH = "files"
TEMPLATE_PATH = "template"


def upload_file(file: UploadFile, path: str, id: str) -> Path:
    """Save uploaded file under files/<id>/. Returns the saved path.

    The client-supplied filename is sanitized to its basename to prevent path
    traversal (e.g. '../../etc/passwd.jpg' → 'passwd.jpg'). The upload size is
    bounded by ensure_upload_size before any bytes are copied."""
    from controller._upload_limits import ensure_upload_size

    ensure_upload_size(file)
    safe_id   = Path(id).name or "anon"
    safe_name = Path(file.filename or "upload.bin").name or "upload.bin"
    dest_dir  = Path(path) / safe_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name
    with dest.open(mode="wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    log.debug("file saved: %s", dest)
    return dest


def verify(files: list[UploadFile], id: str, document_type: str | None = None) -> BaseVerifyResponse:
    """Run the full pipeline on the uploaded files of a single document."""
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    paths: List[Path] = [upload_file(file=f, path=FILES_PATH, id=id) for f in files]

    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    metadata_reports = metadata.extract(paths)
    timings["metadata"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    tampering_reports = tampering.analyze(paths, output_subdir=id)
    timings["tampering"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    processed_pages = preprocessor.process(paths, output_subdir=id)
    timings["preprocessor"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    # ocr_results = ocr.extract(processed_pages, output_subdir=id) # De la branch Andy
    ocr_paths = [
        save_image(page, Path(FILES_PATH) / id / "processed")
        for page in processed_pages
    ]
    ocr_results = ocr.process(ocr_paths, document_type=document_type)
    timings["ocr"] = int((time.perf_counter() - t0) * 1000)

    risk = policy.compute(
        metadata_reports,
        tampering_reports,
        processed_pages,
        ocr_results,
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


# Legacy upload_template removed. Template generation now goes through
# template_controller.generate_template + template_controller.confirm_template.
