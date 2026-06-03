import json
from pathlib import Path

from service.template_ocr.schema import FieldSpec, Template
from service.template_ocr.schema import load_template as _adapter_load


def load_template(templates_dir: str, document_type: str) -> Template | None:
    path = Path(templates_dir) / f"{document_type}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _adapter_load(raw)


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
