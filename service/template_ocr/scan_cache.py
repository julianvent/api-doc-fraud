"""Temporary image storage between /templates/generate and /templates/confirm.

A generate request stores the uploaded file under a UUID with its real
extension (e.g. uuid.pdf, uuid.jpg) so downstream consumers — the preprocessor
in particular — can decide PDF vs image based on the path. Old files are
purged lazily on every load() / save() call.

For the dots mode, two additional files are stored alongside the image:
  {uuid}.elements.json  — DotsOCR element list with normalised bboxes
  {uuid}_preprocessed.png — the preprocessed image array (PNG)
These are cleaned up at confirm time or when the TTL expires.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional


_DEFAULT_DIR = Path(os.getenv("TEMPLATE_SCAN_CACHE_DIR", "tmp/template_scans"))
_DEFAULT_TTL = int(os.getenv("TEMPLATE_SCAN_CACHE_TTL", "3600"))
_DEFAULT_EXT = "jpg"


def _cache_dir() -> Path:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DIR


def _safe_extension(extension: Optional[str]) -> str:
    """Reject anything that isn't a simple alphanumeric extension to avoid path tricks."""
    if not extension:
        return _DEFAULT_EXT
    ext = extension.strip().lower().lstrip(".")
    if not ext or not ext.isalnum() or len(ext) > 6:
        return _DEFAULT_EXT
    return ext


def _resolve_path(generate_id: str) -> Optional[Path]:
    """Find the cached file for this generate_id by globbing for {uuid}.*"""
    try:
        safe = uuid.UUID(generate_id)
    except (ValueError, TypeError):
        return None
    matches = list(_cache_dir().glob(f"{safe}.*"))
    return matches[0] if matches else None


def save(image_bytes: bytes, extension: str = _DEFAULT_EXT) -> str:
    purge_expired()
    generate_id = str(uuid.uuid4())
    ext = _safe_extension(extension)
    path = _cache_dir() / f"{generate_id}.{ext}"
    path.write_bytes(image_bytes)
    return generate_id


def load(generate_id: str) -> Optional[bytes]:
    """Returns the raw bytes (image or PDF) for this generate_id, or None."""
    purge_expired()
    path = _resolve_path(generate_id)
    if path is None or not path.exists():
        return None
    return path.read_bytes()


def path_for(generate_id: str) -> Optional[Path]:
    """Returns the on-disk path with its real extension (for the preprocessor)."""
    purge_expired()
    return _resolve_path(generate_id)


def extension_for(generate_id: str) -> Optional[str]:
    """Returns the file extension (without dot) — e.g. 'pdf', 'jpg'."""
    path = _resolve_path(generate_id)
    if path is None:
        return None
    return path.suffix.lstrip(".").lower() or None


def delete(generate_id: str) -> None:
    path = _resolve_path(generate_id)
    if path is not None and path.exists():
        path.unlink()


def purge_expired(ttl_seconds: int = _DEFAULT_TTL) -> int:
    """Remove cache files older than ttl_seconds. Returns count purged."""
    if not _DEFAULT_DIR.exists():
        return 0
    now    = time.time()
    cutoff = now - ttl_seconds
    purged = 0
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


# ─────────────────────────────────────────────────────── dots-mode helpers


def _elements_path(generate_id: str) -> Optional[Path]:
    try:
        safe = uuid.UUID(generate_id)
    except (ValueError, TypeError):
        return None
    # Underscore prefix so this file is NOT picked up by _resolve_path's
    # glob pattern "{uuid}.*", which only matches files named "{uuid}.ext".
    return _cache_dir() / f"{safe}_elements.json"


def _preprocessed_path(generate_id: str) -> Optional[Path]:
    try:
        safe = uuid.UUID(generate_id)
    except (ValueError, TypeError):
        return None
    return _cache_dir() / f"{safe}_preprocessed.png"


def save_elements(generate_id: str, elements: list[dict]) -> None:
    path = _elements_path(generate_id)
    if path is None:
        return
    path.write_text(json.dumps(elements, ensure_ascii=False), encoding="utf-8")


def load_elements(generate_id: str) -> Optional[list[dict]]:
    path = _elements_path(generate_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_preprocessed_image(generate_id: str, img: "np.ndarray") -> Optional[Path]:
    """Save the preprocessed (numpy RGB) image to cache as PNG. Returns the path."""
    import cv2
    import numpy as np

    path = _preprocessed_path(generate_id)
    if path is None:
        return None
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    cv2.imwrite(str(path), bgr)
    return path


def path_for_preprocessed(generate_id: str) -> Optional[Path]:
    """Returns the on-disk path of the cached preprocessed image, or None."""
    path = _preprocessed_path(generate_id)
    if path is None or not path.exists():
        return None
    return path


def delete_dots_cache(generate_id: str) -> None:
    """Remove both auxiliary dots-mode files for this generate_id."""
    for p in [_elements_path(generate_id), _preprocessed_path(generate_id)]:
        if p is not None and p.exists():
            p.unlink(missing_ok=True)
