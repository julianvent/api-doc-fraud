"""Anomaly rules. Each rule asks a contradiction or impossibility question.

Categories:
    A — internal contradiction      field X disagrees with field Y
    B — claim vs reality            declared value disagrees with measurable file
    C — temporal impossibility      dates outside physically valid range
    D — structural anomaly          structures absent in legitimate files
    E — self-declaration            file explicitly declares synthetic origin / signature
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .extractor import MetadataSnapshot
from .whitelists import (
    AI_GENERATION_CHUNK_KEYS,
    ALWAYS_WRITES_MAKERNOTE,
    IMAGE_EDITORS,
    LEAVES_XMP_HISTORY,
    TRUSTED_GOVT,
    contains_any,
)

Severity = Literal["positive", "low", "medium", "high"]
Category = Literal["A", "B", "C", "D", "E", "S"]

_PDF_ERA_YEAR = 1993
_JPEG_ERA_YEAR = 1992
_PDF_INCREMENTAL_THRESHOLD = 3
_SCANNER_SAVE_THRESHOLD = 3


@dataclass(frozen=True)
class Flag:
    code: str
    severity: Severity
    category: Category
    evidence: str


# ── snapshot accessors ───────────────────────────────────────────────────────

def _all_software_identifiers(snap: MetadataSnapshot) -> str:
    if snap.format == "pdf":
        return f"{snap.pdf.get('producer') or ''} {snap.pdf.get('creator') or ''}"
    return f"{snap.exif.get('Software') or ''} {snap.exif.get('Make') or ''}"


def _xmp_has_history(xmp: str | None) -> bool:
    if not xmp:
        return False
    lowered = xmp.lower()
    return "xmpmm:history" in lowered or ("rdf:bag" in lowered and "stevt:" in lowered)


def _count_xmp_saves(xmp: str | None) -> int:
    if not xmp:
        return 0
    lowered = xmp.lower()
    return max(
        lowered.count('stevt:action="saved"'),
        lowered.count("stevt:action='saved'"),
    )


def _ai_chunks_present(snap: MetadataSnapshot) -> list[str]:
    """Return AI-generation chunk keys found in PNG text or PIL info, case-insensitive."""
    targets = {m.lower() for m in AI_GENERATION_CHUNK_KEYS}
    candidates = set(snap.text_chunks.keys()) | set(snap.info_keys)
    return sorted(k for k in candidates if k.lower() in targets)


# ── E. self-declaration ──────────────────────────────────────────────────────

def rule_trusted_signature(snap: MetadataSnapshot) -> Flag | None:
    if snap.format != "pdf" or not snap.pdf.get("has_signature"):
        return None
    match = contains_any(_all_software_identifiers(snap), TRUSTED_GOVT)
    if not match:
        return None
    return Flag(
        code="TRUSTED_SIGNATURE",
        severity="positive",
        category="S",
        evidence=f"PDF signed; producer matches trusted govt issuer: {match!r}",
    )


def rule_ai_self_declaration(snap: MetadataSnapshot) -> Flag | None:
    if snap.c2pa_ai_assertion:
        marker = snap.c2pa_marker or "unspecified"
        return Flag(
            code="AI_SELF_DECLARATION",
            severity="high",
            category="E",
            evidence=f"C2PA manifest declares synthetic origin (marker: {marker!r})",
        )
    chunks = _ai_chunks_present(snap)
    if chunks:
        return Flag(
            code="AI_SELF_DECLARATION",
            severity="high",
            category="E",
            evidence=f"AI generation chunk(s) found in file: {chunks}",
        )
    return None


# ── D. structural anomaly ────────────────────────────────────────────────────

def rule_active_content(snap: MetadataSnapshot) -> Flag | None:
    if snap.format != "pdf" or not snap.pdf.get("javascript_present"):
        return None
    return Flag(
        code="ACTIVE_CONTENT",
        severity="high",
        category="D",
        evidence="PDF contains /JS or /JavaScript objects",
    )


def rule_excessive_pdf_incremental(snap: MetadataSnapshot) -> Flag | None:
    if snap.format != "pdf" or snap.pdf.get("has_signature"):
        return None
    incr = snap.pdf.get("incremental_updates") or 0
    if not isinstance(incr, int) or incr <= _PDF_INCREMENTAL_THRESHOLD:
        return None
    return Flag(
        code="EXCESSIVE_PDF_INCREMENTAL",
        severity="medium",
        category="D",
        evidence=f"PDF has {incr} incremental updates without signature",
    )


# ── B. claim vs reality ──────────────────────────────────────────────────────

def rule_magic_extension_mismatch(snap: MetadataSnapshot) -> Flag | None:
    if snap.detected_format is None or snap.magic_match:
        return None
    return Flag(
        code="MAGIC_EXTENSION_MISMATCH",
        severity="high",
        category="B",
        evidence=(
            f"Extension says {snap.format!r} but magic bytes identify "
            f"{snap.detected_format!r}"
        ),
    )


def rule_dimension_mismatch(snap: MetadataSnapshot) -> Flag | None:
    if snap.format == "pdf":
        return None
    decl_w = snap.exif.get("ExifImageWidth") or snap.exif.get("ImageWidth")
    decl_h = snap.exif.get("ExifImageHeight") or snap.exif.get("ImageLength")
    actual = snap.image.get("size")
    if not (decl_w and decl_h and actual and len(actual) == 2):
        return None
    try:
        if int(decl_w) == int(actual[0]) and int(decl_h) == int(actual[1]):
            return None
    except (ValueError, TypeError):
        return None
    return Flag(
        code="DIMENSION_MISMATCH",
        severity="medium",
        category="B",
        evidence=f"EXIF dims ({decl_w}x{decl_h}) != actual {actual[0]}x{actual[1]}",
    )


# ── A. internal contradiction ────────────────────────────────────────────────

def rule_device_fingerprint_inconsistent(snap: MetadataSnapshot) -> Flag | None:
    if snap.format == "pdf":
        return None
    make = (snap.exif.get("Make") or "").strip()
    if not make or "MakerNote" in snap.exif:
        return None
    if not contains_any(make, ALWAYS_WRITES_MAKERNOTE):
        return None
    return Flag(
        code="DEVICE_FINGERPRINT_INCONSISTENT",
        severity="medium",
        category="A",
        evidence=f"Make={make!r} (always writes MakerNote) but MakerNote absent",
    )


def rule_editor_without_history(snap: MetadataSnapshot) -> Flag | None:
    match = contains_any(_all_software_identifiers(snap), LEAVES_XMP_HISTORY)
    if not match or _xmp_has_history(snap.xmp):
        return None
    return Flag(
        code="EDITOR_WITHOUT_HISTORY",
        severity="medium",
        category="A",
        evidence=f"Software={match!r} (always writes xmpMM:History) but history absent",
    )


def rule_editor_software_present(snap: MetadataSnapshot) -> Flag | None:
    """Image editor declared as Software/producer of an ID document."""
    match = contains_any(_all_software_identifiers(snap), IMAGE_EDITORS)
    if not match:
        return None
    return Flag(
        code="EDITOR_SOFTWARE_PRESENT",
        severity="medium",
        category="A",
        evidence=f"Image editor declared in Software/producer: {match!r}",
    )


def rule_missing_camera_exif(snap: MetadataSnapshot) -> Flag | None:
    """JPEG without any camera/edit provenance signals.

    A JPG of a real document is normally produced by a camera (leaves EXIF
    DateTimeOriginal / Make / Model) or by an editor (leaves Software / XMP).
    A file with none of those is anomalous — typically a re-encoded crop or
    AI-generated raster saved as bare JFIF.
    """
    if snap.format not in {"jpg", "jpeg"}:
        return None
    has_camera_exif = any(
        snap.exif.get(k) for k in ("DateTimeOriginal", "Make", "Model")
    )
    has_software = bool(snap.exif.get("Software"))
    has_xmp = bool(snap.xmp)
    if has_camera_exif or has_software or has_xmp:
        return None
    return Flag(
        code="MISSING_CAMERA_EXIF",
        severity="medium",
        category="D",
        evidence="JPEG lacks DateTimeOriginal/Make/Model/Software/XMP (bare JFIF)",
    )


def rule_scanner_with_edit_history(snap: MetadataSnapshot) -> Flag | None:
    if snap.format != "pdf":
        return None
    producer = snap.pdf.get("producer") or ""
    if "scan" not in producer.lower():
        return None
    saves = _count_xmp_saves(snap.xmp)
    if saves < _SCANNER_SAVE_THRESHOLD:
        return None
    return Flag(
        code="SCANNER_WITH_EDIT_HISTORY",
        severity="medium",
        category="A",
        evidence=f"Producer={producer!r} (scanner) but XMP shows {saves} save actions",
    )


def rule_date_inconsistency(snap: MetadataSnapshot) -> Flag | None:
    if snap.format != "pdf":
        return None
    cd = _parse_pdf_date(snap.pdf.get("creation_date"))
    md = _parse_pdf_date(snap.pdf.get("mod_date"))
    if not (cd and md and md < cd):
        return None
    return Flag(
        code="DATE_INCONSISTENCY",
        severity="high",
        category="A",
        evidence=f"PDF ModDate ({md}) precedes CreationDate ({cd})",
    )


# ── C. temporal impossibility ────────────────────────────────────────────────

def rule_temporal_anomaly(snap: MetadataSnapshot) -> Flag | None:
    anomalies = (
        _pdf_temporal_anomalies(snap) if snap.format == "pdf"
        else _exif_temporal_anomalies(snap)
    )
    if not anomalies:
        return None
    severity = _temporal_severity(anomalies)
    return Flag(
        code="TEMPORAL_ANOMALY",
        severity=severity,
        category="C",
        evidence="; ".join(anomalies),
    )


def _pdf_temporal_anomalies(snap: MetadataSnapshot) -> list[str]:
    out: list[str] = []
    now = datetime.now()
    for label, raw in (("creation_date", snap.pdf.get("creation_date")),
                       ("mod_date", snap.pdf.get("mod_date"))):
        ts = _parse_pdf_date(raw)
        if ts is None:
            continue
        if ts > now:
            out.append(f"{label} in future ({ts})")
        elif ts.year < _PDF_ERA_YEAR:
            out.append(f"{label} before PDF era ({ts})")
    return out


def _exif_temporal_anomalies(snap: MetadataSnapshot) -> list[str]:
    out: list[str] = []
    now = datetime.now()
    for label, raw in (("DateTimeOriginal", snap.exif.get("DateTimeOriginal")),
                       ("DateTime", snap.exif.get("DateTime"))):
        ts = _parse_exif_date(raw)
        if ts is None:
            continue
        if ts > now:
            out.append(f"{label} in future ({ts})")
        elif ts.year < _JPEG_ERA_YEAR:
            out.append(f"{label} before JPEG era ({ts})")
    return out


def _temporal_severity(anomalies: list[str]) -> Severity:
    if any("future" in a or "before" in a for a in anomalies):
        return "high"
    return "medium" if len(anomalies) >= 2 else "low"


# ── date parsers ─────────────────────────────────────────────────────────────

def _parse_pdf_date(val: str | None) -> datetime | None:
    if not val:
        return None
    s = str(val).strip()
    if s.startswith("D:"):
        s = s[2:]
    digits = "".join(c for c in s if c.isdigit())[:14]
    if len(digits) < 8:
        return None
    padded = (digits + "000000000000000")[:14]
    try:
        return datetime.strptime(padded, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_exif_date(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.strptime(str(val)[:19], "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


# ── runner ───────────────────────────────────────────────────────────────────

_RULES = (
    rule_trusted_signature,
    rule_ai_self_declaration,
    rule_active_content,
    rule_excessive_pdf_incremental,
    rule_magic_extension_mismatch,
    rule_dimension_mismatch,
    rule_device_fingerprint_inconsistent,
    rule_editor_without_history,
    rule_editor_software_present,
    rule_missing_camera_exif,
    rule_scanner_with_edit_history,
    rule_date_inconsistency,
    rule_temporal_anomaly,
)


def evaluate(snap: MetadataSnapshot) -> list[Flag]:
    """Run every rule and return the flags that fired."""
    flags: list[Flag] = []
    for rule in _RULES:
        result = rule(snap)
        if result is not None:
            flags.append(result)
    return flags
