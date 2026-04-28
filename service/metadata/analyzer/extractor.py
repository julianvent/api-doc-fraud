"""Raw metadata collection. Extracts every available signal without judgment."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz
from PIL import ExifTags, Image

from .whitelists import (
    C2PA_AI_ACTION_MARKERS,
    C2PA_PRESENCE_MARKERS,
    C2PA_SYNTHETIC_SOURCE_TYPES,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_EXTS = {".pdf"}

_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"%PDF", "pdf"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

_EXT_TO_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff"}

_C2PA_SCAN_LIMIT = 4 * 1024 * 1024
_TEXT_VALUE_LIMIT = 800
_FREEFORM_TEXT_KEYS = ("comment", "Description", "Comment")


@dataclass
class MetadataSnapshot:
    source: str
    format: str
    file_size: int = 0
    detected_format: str | None = None
    magic_match: bool = True
    exif: dict = field(default_factory=dict)
    xmp: str | None = None
    pdf: dict = field(default_factory=dict)
    image: dict = field(default_factory=dict)
    info_keys: list[str] = field(default_factory=list)
    text_chunks: dict = field(default_factory=dict)
    icc_profile_present: bool = False
    quant_table_count: int = 0
    c2pa_present: bool = False
    c2pa_ai_assertion: bool = False
    c2pa_marker: str | None = None
    warnings: list[str] = field(default_factory=list)


class MetadataExtractor:
    """Format-aware extractor producing a uniform MetadataSnapshot."""

    def extract(self, path: str | Path) -> MetadataSnapshot:
        path = Path(path)
        ext = path.suffix.lower()

        if ext in _PDF_EXTS:
            snap = self._extract_pdf(path)
        elif ext in _IMAGE_EXTS:
            snap = self._extract_image(path)
        else:
            snap = MetadataSnapshot(source=str(path), format="unknown")
            snap.warnings.append(f"Unsupported extension: {ext}")

        snap.file_size = _safe_filesize(path)
        snap.detected_format, snap.magic_match = _detect_magic(path, ext)
        snap.c2pa_present, snap.c2pa_ai_assertion, snap.c2pa_marker = _scan_c2pa(path)
        return snap

    def _extract_image(self, path: Path) -> MetadataSnapshot:
        snap = MetadataSnapshot(source=str(path), format=path.suffix.lower().lstrip("."))
        try:
            with Image.open(path) as img:
                img.load()
                snap.image = {
                    "mode": img.mode,
                    "size": list(img.size),
                    "format": img.format,
                    "has_thumbnail": bool(img.info.get("thumbnail")),
                }
                snap.info_keys = sorted(str(k) for k in img.info.keys())
                snap.icc_profile_present = bool(img.info.get("icc_profile"))

                quant = getattr(img, "quantization", None)
                if isinstance(quant, dict):
                    snap.quant_table_count = len(quant)

                snap.exif = _read_exif(img)
                snap.xmp = _read_xmp(img)
                snap.text_chunks = _read_text_chunks(img)
        except Exception as exc:
            snap.warnings.append(f"Image read failed: {exc}")
        return snap

    def _extract_pdf(self, path: Path) -> MetadataSnapshot:
        snap = MetadataSnapshot(source=str(path), format="pdf")
        try:
            doc = fitz.open(str(path))
            try:
                info = doc.metadata or {}
                snap.pdf = {
                    "producer": info.get("producer"),
                    "creator": info.get("creator"),
                    "creation_date": info.get("creationDate"),
                    "mod_date": info.get("modDate"),
                    "title": info.get("title"),
                    "author": info.get("author"),
                    "subject": info.get("subject"),
                    "keywords": info.get("keywords"),
                    "page_count": doc.page_count,
                    "is_encrypted": doc.is_encrypted,
                    "has_signature": _pdf_has_signature(doc),
                    "incremental_updates": _pdf_eof_count(path),
                    "javascript_present": _pdf_has_javascript(doc),
                }
                xmp = doc.get_xml_metadata()
                if xmp:
                    snap.xmp = xmp
            finally:
                doc.close()
        except Exception as exc:
            snap.warnings.append(f"PDF read failed: {exc}")
        return snap


# ── image helpers ────────────────────────────────────────────────────────────

def _read_exif(img: Image.Image) -> dict:
    raw = img.getexif()
    if not raw:
        return {}
    return {
        ExifTags.TAGS.get(tag, f"tag_{tag}"): _to_jsonable(val)
        for tag, val in raw.items()
    }


def _read_xmp(img: Image.Image) -> str | None:
    raw = img.info.get("xmp")
    if not raw:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _read_text_chunks(img: Image.Image) -> dict[str, str]:
    chunks: dict[str, str] = {}
    text = getattr(img, "text", None)
    if isinstance(text, dict):
        for k, v in text.items():
            chunks[str(k)] = "" if v is None else str(v)[:_TEXT_VALUE_LIMIT]
    for k in _FREEFORM_TEXT_KEYS:
        if k in chunks:
            continue
        val = img.info.get(k)
        if not val:
            continue
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8", errors="replace")
        chunks[k] = str(val)[:_TEXT_VALUE_LIMIT]
    return chunks


# ── file-level helpers ───────────────────────────────────────────────────────

def _safe_filesize(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _detect_magic(path: Path, ext: str) -> tuple[str | None, bool]:
    """Return (detected_format, extension_matches). Magic-vs-extension check."""
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return None, True

    detected: str | None = None
    for sig, fmt in _MAGIC_SIGNATURES:
        if head.startswith(sig):
            detected = fmt
            break
    if head[:4] == b"RIFF" and len(head) >= 12 and head[8:12] == b"WEBP":
        detected = "webp"

    if detected is None:
        return None, True
    ext_norm = _EXT_TO_FORMAT.get(ext, ext.lstrip("."))
    return detected, detected == ext_norm


def _scan_c2pa(path: Path) -> tuple[bool, bool, str | None]:
    """Detect C2PA manifest and return (present, ai_assertion, matched_marker)."""
    try:
        with path.open("rb") as f:
            data = f.read(_C2PA_SCAN_LIMIT)
    except OSError:
        return False, False, None

    if not any(m in data for m in C2PA_PRESENCE_MARKERS):
        return False, False, None

    for marker in C2PA_AI_ACTION_MARKERS + C2PA_SYNTHETIC_SOURCE_TYPES:
        if marker in data:
            return True, True, marker.decode("ascii", errors="replace")
    return True, False, None


# ── pdf helpers ──────────────────────────────────────────────────────────────

_PDF_WIDGET_TYPE_SIGNATURE = 6


def _pdf_has_signature(doc: fitz.Document) -> bool:
    try:
        for page in doc:
            for widget in page.widgets() or []:
                if getattr(widget, "field_type", None) == _PDF_WIDGET_TYPE_SIGNATURE:
                    return True
    except Exception:
        pass
    return False


def _pdf_has_javascript(doc: fitz.Document) -> bool:
    try:
        for xref in range(1, doc.xref_length()):
            obj = doc.xref_object(xref) or ""
            if "/JavaScript" in obj or "/JS" in obj:
                return True
    except Exception:
        pass
    return False


def _pdf_eof_count(path: Path) -> int:
    try:
        return path.read_bytes().count(b"%%EOF")
    except OSError:
        return 0


# ── serialization ────────────────────────────────────────────────────────────

def _to_jsonable(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, (tuple, list)):
        return [_to_jsonable(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    return val


def _main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/visa.pdf"
    snap = MetadataExtractor().extract(path)
    print(json.dumps(asdict(snap), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    _main()
