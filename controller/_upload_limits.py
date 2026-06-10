"""Shared upload-size limit enforcement for FastAPI endpoints.

Limits apply to BOTH `/v1/verify` (document images) and `/v1/templates/generate`
(template sample image). The limit is configurable via the MAX_UPLOAD_BYTES
environment variable (default: 10 MB)."""

from __future__ import annotations

import os

from fastapi import HTTPException, UploadFile


_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_BYTES   = int(os.getenv("MAX_UPLOAD_BYTES", str(_DEFAULT_MAX_BYTES)))


def ensure_upload_size(file: UploadFile, *, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Reject uploads larger than max_bytes by raising HTTPException 413.

    Uses Starlette's UploadFile.size when available (set after headers parsing).
    Falls back to a no-op when the size attribute is missing (very old runtimes
    or in-memory mocks); in that case the upstream code reading the bytes is
    still bounded by `max_bytes` via read_within_limit."""
    size = getattr(file, "size", None)
    if size is not None and size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file '{file.filename}' is {size} bytes, "
                f"exceeds the limit of {max_bytes} bytes"
            ),
        )


def read_within_limit(file: UploadFile, *, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read up to max_bytes+1 from file; if it exceeds max_bytes, raise 413.

    Guards against missing/lying Content-Length headers."""
    ensure_upload_size(file, max_bytes=max_bytes)
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file body exceeds the limit of {max_bytes} bytes",
        )
    return data
