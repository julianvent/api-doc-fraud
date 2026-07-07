import re
from datetime import datetime
from typing import Any


_COMMON_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b. %Y",
    "%Y/%m/%d",
    "%d %m %Y",
]


def _parse_date(value: Any, fmt: str | None = None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = [fmt] if fmt else _COMMON_DATE_FORMATS
    for f in candidates:
        if not f:
            continue
        try:
            return datetime.strptime(text, f)
        except ValueError:
            continue
    return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _check_required(field: str, value: Any, rule: dict) -> list[dict]:
    if not rule.get("required"):
        return []
    if _is_empty(value):
        return [{"field": field, "description": f"{field} is required but missing"}]
    return []


def _check_regex(field: str, value: Any, rule: dict) -> list[dict]:
    pattern = rule.get("regex")
    if not pattern or _is_empty(value):
        return []
    if not isinstance(value, str):
        return []
    if not re.fullmatch(pattern, value.strip()):
        return [{
            "field"      : field,
            "description": f"{field} value '{value}' does not match pattern '{pattern}'",
        }]
    return []


def _check_date(field: str, value: Any, rule: dict, all_fields: dict) -> list[dict]:
    if rule.get("type") != "date":
        return []
    if _is_empty(value):
        return []

    fmt    = rule.get("format")
    parsed = _parse_date(value, fmt)
    if parsed is None:
        desc = (
            f"{field} value '{value}' cannot be parsed as date with format '{fmt}'"
            if fmt else
            f"{field} value '{value}' cannot be parsed as date"
        )
        return [{"field": field, "description": desc}]

    issues = []
    now    = datetime.now()

    if rule.get("must_be_past") and parsed >= now:
        issues.append({
            "field"      : field,
            "description": f"{field} '{value}' must be in the past",
        })

    if rule.get("must_be_future") and parsed <= now:
        issues.append({
            "field"      : field,
            "description": f"{field} '{value}' must be in the future",
        })

    for op_key, op in (("before", "<"), ("after", ">")):
        other_field = rule.get(op_key)
        if not other_field:
            continue
        other_value = all_fields.get(other_field)
        if _is_empty(other_value):
            continue
        other_parsed = _parse_date(other_value, fmt) or _parse_date(other_value)
        if other_parsed is None:
            continue
        valid = (parsed < other_parsed) if op == "<" else (parsed > other_parsed)
        if not valid:
            relation = "before" if op == "<" else "after"
            issues.append({
                "field"      : field,
                "description": f"{field} '{value}' must be {relation} {other_field} '{other_value}'",
            })

    return issues


def apply_rules(fields: dict[str, Any],
                field_rules: dict[str, dict] | None) -> list[dict]:
    if not field_rules or not isinstance(field_rules, dict):
        return []

    issues: list[dict] = []
    for field, rule in field_rules.items():
        if not isinstance(rule, dict):
            continue
        value = fields.get(field)
        issues.extend(_check_required(field, value, rule))
        issues.extend(_check_regex(field, value, rule))
        issues.extend(_check_date(field, value, rule, fields))

    return issues
