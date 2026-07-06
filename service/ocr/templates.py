import json
from pathlib import Path
from typing import Optional

from service.template_ocr.schema import FieldSpec, Template
from service.template_ocr.schema import load_template as _adapter_load


def _candidate_paths(
    templates_dir: Path,
    document_type: str,
    country_iso: Optional[str],
    edition: Optional[int],
) -> list[Path]:
    """Return JSON file candidates for a given (document_type, country_iso, edition).
    The newer convention is `{document_type}_{country_iso or 'ANY'}_{edition}.json`.
    Legacy single-file convention `{document_type}.json` is kept as a fallback.
    Files are ranked from most-specific to least-specific."""
    candidates: list[Path] = []

    iso  = (country_iso or "any").upper()
    glob = sorted(templates_dir.glob(f"{document_type}_*_*.json"))

    if edition is not None:
        # Exact match: doc_type + country_iso + edition
        candidates.append(templates_dir / f"{document_type}_{iso}_{edition}.json")
        # Same edition, any country
        candidates.append(templates_dir / f"{document_type}_ANY_{edition}.json")

    # Latest edition for the requested country
    matching_country = [
        p for p in glob
        if _parse_filename(p.stem) and _parse_filename(p.stem)[1] == (iso if iso != "ANY" else None)
    ]
    matching_country.sort(key=lambda p: _parse_filename(p.stem)[2], reverse=True)
    candidates.extend(matching_country)

    # Latest edition, any country (last resort)
    sorted_any = sorted(
        (p for p in glob if _parse_filename(p.stem)),
        key=lambda p: _parse_filename(p.stem)[2],
        reverse=True,
    )
    candidates.extend(sorted_any)

    # Legacy single-file fallback
    candidates.append(templates_dir / f"{document_type}.json")

    # Dedupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _parse_filename(stem: str) -> Optional[tuple[str, Optional[str], int]]:
    """Parses '{document_type}_{COUNTRY}_{edition}' → (document_type, country_iso, edition)."""
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    doc_type, country, edition_str = parts
    try:
        edition = int(edition_str)
    except ValueError:
        return None
    iso = None if country.upper() == "ANY" else country.upper()
    return doc_type, iso, edition


def load_template(
    document_type: str,
    country_iso: Optional[str] = None,
    edition: Optional[int] = None,
    templates_dir: Optional[str] = None,
) -> Optional[Template]:
    """Load a template JSON from disk.
    Tries in order: exact (doc_type+country_iso+edition) → latest for country →
    latest for any country → legacy {document_type}.json."""
    from service.ocr.models import Config
    base = Path(templates_dir) if templates_dir else Path(Config().templates_dir)

    for path in _candidate_paths(base, document_type, country_iso, edition):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        return _adapter_load(raw)
    return None


def iter_template_fields(template: Template) -> list[FieldSpec]:
    return list(template.fields)


def build_fields_guide(template: Template) -> str:
    if not template.fields:
        return ""

    has_categories = any(f.category for f in template.fields)

    if has_categories:
        groups: dict[str, list[FieldSpec]] = {}
        for f in template.fields:
            groups.setdefault(f.category or "other", []).append(f)
        sections = []
        for category, group in groups.items():
            lines = [f"### {category}"]
            for f in group:
                lines.append(f"- {f.key}: look for label '{f.label}' (value type: {f.type})")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    return "\n".join(
        f"- {f.key}: look for label '{f.label}' (value type: {f.type})"
        for f in template.fields
    )
