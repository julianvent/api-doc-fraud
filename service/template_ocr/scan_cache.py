"""Temporary image storage between /templates/generate and /templates/confirm.

A generate request stores the uploaded image under a UUID, returns the id, and
expects a confirm request to reference the same id within the TTL window. Old
files are purged lazily on every load() / save() call.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional


_DEFAULT_DIR = Path(os.getenv("TEMPLATE_SCAN_CACHE_DIR", "tmp/template_scans"))
_DEFAULT_TTL = int(os.getenv("TEMPLATE_SCAN_CACHE_TTL", "3600"))


def _cache_dir() -> Path:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DIR


def _path_for(generate_id: str) -> Path:
    safe = uuid.UUID(generate_id)
    return _cache_dir() / f"{safe}.bin"


def save(image_bytes: bytes, extension: str = "jpg") -> str:
    purge_expired()
    generate_id = str(uuid.uuid4())
    path = _cache_dir() / f"{generate_id}.bin"
    path.write_bytes(image_bytes)
    if extension:
        ext_path = _cache_dir() / f"{generate_id}.ext"
        ext_path.write_text(extension, encoding="utf-8")
    return generate_id


def load(generate_id: str) -> Optional[bytes]:
    purge_expired()
    try:
        path = _path_for(generate_id)
    except (ValueError, TypeError):
        return None
    if not path.exists():
        return None
    return path.read_bytes()


def extension_for(generate_id: str) -> Optional[str]:
    try:
        ext_path = _cache_dir() / f"{uuid.UUID(generate_id)}.ext"
    except (ValueError, TypeError):
        return None
    if not ext_path.exists():
        return None
    return ext_path.read_text(encoding="utf-8").strip() or None


def delete(generate_id: str) -> None:
    try:
        path = _path_for(generate_id)
    except (ValueError, TypeError):
        return
    if path.exists():
        path.unlink()
    ext_path = _cache_dir() / f"{path.stem}.ext"
    if ext_path.exists():
        ext_path.unlink()


def purge_expired(ttl_seconds: int = _DEFAULT_TTL) -> int:
    """Remove cache files older than ttl_seconds. Returns count purged."""
    if not _DEFAULT_DIR.exists():
        return 0
    now      = time.time()
    purged   = 0
    cutoff   = now - ttl_seconds
    for entry in _DEFAULT_DIR.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                purged += 1
        except FileNotFoundError:
            continue
    return purged


def get_ttl_seconds() -> int:
    return _DEFAULT_TTL
