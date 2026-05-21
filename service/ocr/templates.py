import json
from pathlib import Path


def load_template(templates_dir: str, document_type: str) -> dict | None:
    path = Path(templates_dir) / f"{document_type}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_template_fields(template: dict) -> list[dict]:
    fields = template.get("fields", [])
    if isinstance(fields, list):
        return [f for f in fields if isinstance(f, dict)]
    if isinstance(fields, dict):
        out = []
        for group in fields.values():
            if isinstance(group, list):
                out.extend(f for f in group if isinstance(f, dict))
        return out
    return []


def build_fields_guide(template: dict) -> str:
    fields = template.get("fields", [])
    if isinstance(fields, dict):
        sections = []
        for category, group in fields.items():
            if not isinstance(group, list) or not group:
                continue
            lines = [f"### {category}"]
            for f in group:
                if not isinstance(f, dict):
                    continue
                key       = f.get("key", "")
                label     = f.get("label", "")
                type_hint = f.get("type", "")
                lines.append(f"- {key}: look for label '{label}' (value type: {type_hint})")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    if isinstance(fields, list):
        lines = []
        for f in fields:
            if not isinstance(f, dict):
                continue
            key       = f.get("key", "")
            label     = f.get("label", "")
            type_hint = f.get("type", "")
            lines.append(f"- {key}: look for label '{label}' (value type: {type_hint})")
        return "\n".join(lines)

    return ""
