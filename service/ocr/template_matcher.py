import json
from pathlib import Path
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import TextLine


TEMPLATES_DIR         = Path(__file__).parent / "templates"
BBOX_TOLERANCE        = 0.04
MATCH_THRESHOLD       = 0.80
LABEL_FUZZY_THRESHOLD = 70


@dataclass
class MatchResult:
    matched          : bool
    match_score      : float
    document_type    : str
    document_name    : str
    field_lines      : dict[str, list[TextLine]]
    unmatched_fields : list[str]


def load_template(document_type: str) -> dict | None:
    candidates = sorted(TEMPLATES_DIR.glob(f"{document_type}*.json"))
    for path in candidates:
        if path.stem == document_type:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    if candidates:
        with open(candidates[0], "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_templates(document_type: str) -> list[dict]:
    candidates = sorted(TEMPLATES_DIR.glob(f"{document_type}*.json"))
    templates = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                templates.append(json.load(f))
        except Exception:
            continue
    return templates


def _line_center(line: TextLine) -> tuple[float, float] | None:
    if line.bbox is None or len(line.bbox) == 0:
        return None
    xs = [pt[0] for pt in line.bbox]
    ys = [pt[1] for pt in line.bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _line_in_region(line: TextLine, region: dict, tolerance: float = BBOX_TOLERANCE) -> bool:
    center = _line_center(line)
    if center is None:
        return False
    cx, cy = center
    return (
        region["x1"] - tolerance <= cx <= region["x2"] + tolerance and
        region["y1"] - tolerance <= cy <= region["y2"] + tolerance
    )


def _find_lines_in_region(lines: list[TextLine], region: dict, tolerance: float = BBOX_TOLERANCE) -> list[TextLine]:
    return [l for l in lines if _line_in_region(l, region, tolerance)]


_ROW_GROUP_THRESHOLD = 0.03


def _cy(line: TextLine) -> float:
    ys = [pt[1] for pt in line.bbox]
    return (min(ys) + max(ys)) / 2


def _first_row(lines: list[TextLine]) -> list[TextLine]:
    if not lines:
        return lines
    sorted_lines = sorted(lines, key=_cy)
    base_y = _cy(sorted_lines[0])
    return [l for l in sorted_lines if _cy(l) - base_y <= _ROW_GROUP_THRESHOLD]


def _find_label_line(lines: list[TextLine], label_latin: str) -> TextLine | None:
    if not label_latin:
        return None
    matches = [(l, fuzz.partial_ratio(label_latin, l.text)) for l in lines]
    matches = [(l, s) for l, s in matches if s >= LABEL_FUZZY_THRESHOLD]
    if not matches:
        return None
    top_score = max(s for _, s in matches)
    top_matches = [l for l, s in matches if s == top_score]
    return min(top_matches, key=_cy)


def _latin_part(label: str) -> str:
    if "/" in label:
        label = label.split("/", 1)[1]
    return "".join(c for c in label if ord(c) < 0x0900).strip()


def match(lines: list[TextLine], template: dict) -> MatchResult:
    fields        = template.get("fields", [])
    document_type = template.get("document_type", "unknown")
    document_name = template.get("document_name", "unknown")

    ocr_text = " ".join(l.text for l in lines)

    field_lines      : dict[str, list[TextLine]] = {}
    unmatched_fields : list[str]                 = []
    field_hits       = 0

    all_label_ids: set[int] = set()
    for fd in fields:
        lr = fd.get("label_region")
        if lr:
            all_label_ids.update(id(l) for l in _find_lines_in_region(lines, lr, tolerance=0.0))

    for field_def in fields:
        key            = field_def.get("key", "")
        expected_label = field_def.get("label", "")
        value_region   = field_def.get("value_region")

        label_latin = _latin_part(expected_label)
        score       = fuzz.partial_ratio(label_latin, ocr_text) if label_latin else 0
        label_found = score >= LABEL_FUZZY_THRESHOLD

        if label_found:
            field_hits += 1
            value_lines = _find_lines_in_region(lines, value_region) if value_region else []
            value_lines = [l for l in value_lines if id(l) not in all_label_ids]

            label_line = _find_label_line(lines, label_latin)
            if label_line is not None:
                anchor_y = _cy(label_line)
                value_lines = [l for l in value_lines if _cy(l) > anchor_y]

            value_lines = _first_row(value_lines)
            field_lines[key] = value_lines
        else:
            unmatched_fields.append(key)
            field_lines[key] = []

    match_score = field_hits / len(fields) if fields else 0.0

    return MatchResult(
        matched          = match_score >= MATCH_THRESHOLD,
        match_score      = round(match_score, 3),
        document_type    = document_type,
        document_name    = document_name,
        field_lines      = field_lines,
        unmatched_fields = unmatched_fields
    )
