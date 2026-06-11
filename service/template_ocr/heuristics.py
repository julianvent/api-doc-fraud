"""Deterministic field suggestion heuristics — no VLM/LLM.

Three sources of suggestions, ranked by confidence:

  1. MRZ → fields directly decoded from the MRZ block (high confidence).
  2. Regex → tokens matching well-known patterns (dates, currency, email,
     CURP, RFC, etc.) with the nearest non-numeric line as candidate label.
  3. Spatial match → given a list of user-provided expected fields (key + label),
     locate each label in the OCR lines via fuzzy match, then propose the
     adjacent line as the value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from rapidfuzz import fuzz

from service.ocr.models import MRZResult, TextLine


_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("birth_date",      "date",        re.compile(r"\b\d{2}[/.-]\d{2}[/.-]\d{4}\b")),
    ("amount_due",      "currency",    re.compile(r"\$\s?[\d,]+\.\d{2}")),
    ("email",           "text",        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone",           "text",        re.compile(r"\b\d{2}\s?\d{4}\s?\d{4}\b")),
    ("personal_id",     "alphanumeric", re.compile(r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]\d\b")),  # CURP
    ("rfc",             "alphanumeric", re.compile(r"\b[A-Z]{3,4}\d{6}[A-Z\d]{3}\b")),
    ("document_number", "alphanumeric", re.compile(r"\b[A-Z]\d{8,9}\b")),
]

_FUZZY_THRESHOLD_HIGH   = 90
_FUZZY_THRESHOLD_MEDIUM = 70

_LABEL_HAS_DIGITS = re.compile(r"\d")


@dataclass
class Suggestion:
    key            : str
    label          : str
    type           : str
    value_preview  : Optional[str]
    label_line_id  : Optional[int]
    value_line_ids : list[int]
    confidence     : str   # "high" | "medium" | "low"
    source         : str   # "mrz" | "regex" | "spatial_match"

    def to_dict(self) -> dict:
        return {
            "key":             self.key,
            "label":           self.label,
            "type":            self.type,
            "value_preview":   self.value_preview,
            "label_line_id":   self.label_line_id,
            "value_line_ids":  list(self.value_line_ids),
            "confidence":      self.confidence,
            "source":          self.source,
        }


def _bbox_center(bbox) -> Optional[tuple[float, float]]:
    if bbox is None:
        return None
    try:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    except Exception:
        return None


def _find_label_for(value_line_idx: int, lines: list[TextLine]) -> Optional[int]:
    """Find the nearest line to the LEFT or ABOVE that has no digits — likely the label."""
    value_center = _bbox_center(lines[value_line_idx].bbox)
    if value_center is None:
        return None
    vx, vy = value_center

    best_idx       = None
    best_distance  = float("inf")
    for i, line in enumerate(lines):
        if i == value_line_idx:
            continue
        if _LABEL_HAS_DIGITS.search(line.text):
            continue
        c = _bbox_center(line.bbox)
        if c is None:
            continue
        cx, cy = c
        if cx <= vx and abs(cy - vy) < 40:
            distance = vx - cx
        elif cy <= vy and abs(cx - vx) < 200:
            distance = (vy - cy) * 1.5
        else:
            continue
        if distance < best_distance:
            best_distance = distance
            best_idx      = i

    return best_idx


def suggest_from_mrz(mrz: Optional[MRZResult]) -> list[Suggestion]:
    if mrz is None:
        return []
    out: list[Suggestion] = []
    fields: list[tuple[str, str, str, Optional[str]]] = [
        ("surname",         "Surname",         "text",         mrz.surname),
        ("given_names",     "Given names",     "text",         mrz.given_names),
        ("document_number", "Document number", "alphanumeric", mrz.number),
        ("birth_date",      "Date of birth",   "date",         mrz.birth_date),
        ("expiry_date",     "Date of expiry",  "date",         mrz.expiry_date),
        ("sex",             "Sex",             "single_letter", mrz.sex),
        ("nationality",     "Nationality",     "code",         mrz.country),
    ]
    for key, label, ftype, value in fields:
        if not value:
            continue
        out.append(Suggestion(
            key            = key,
            label          = label,
            type           = ftype,
            value_preview  = value,
            label_line_id  = None,
            value_line_ids = [],
            confidence     = "high",
            source         = "mrz",
        ))
    return out


def suggest_from_regex(lines: list[TextLine]) -> list[Suggestion]:
    out: list[Suggestion] = []
    seen_keys: set[str] = set()

    for idx, line in enumerate(lines):
        text = line.text
        for key, ftype, pattern in _PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            if key in seen_keys:
                # already proposed this key from an earlier line; skip duplicates
                continue
            seen_keys.add(key)

            label_idx  = _find_label_for(idx, lines)
            label_text = lines[label_idx].text if label_idx is not None else key.replace("_", " ").title()

            out.append(Suggestion(
                key            = key,
                label          = label_text,
                type           = ftype,
                value_preview  = match.group(0),
                label_line_id  = label_idx,
                value_line_ids = [idx],
                confidence     = "medium",
                source         = "regex",
            ))
            break
    return out


def suggest_from_expected(
    expected_fields: Iterable[dict],
    lines: list[TextLine],
) -> list[Suggestion]:
    """For each expected field {key,label}, fuzzy-match against OCR text and
    propose the adjacent line as value."""
    out: list[Suggestion] = []

    for spec in expected_fields:
        key   = str(spec.get("key", "")).strip()
        label = str(spec.get("label", "")).strip()
        ftype = str(spec.get("type", "text")).strip() or "text"
        if not key or not label:
            continue

        best_idx    = None
        best_score  = 0.0
        for i, line in enumerate(lines):
            if not line.text:
                continue
            score = fuzz.partial_ratio(label.lower(), line.text.lower())
            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx is None or best_score < _FUZZY_THRESHOLD_MEDIUM:
            continue

        value_idx = _find_value_for(best_idx, lines)
        if value_idx is None:
            continue

        if best_score >= _FUZZY_THRESHOLD_HIGH:
            confidence = "high"
        else:
            confidence = "medium"

        out.append(Suggestion(
            key            = key,
            label          = label,
            type           = ftype,
            value_preview  = lines[value_idx].text,
            label_line_id  = best_idx,
            value_line_ids = [value_idx],
            confidence     = confidence,
            source         = "spatial_match",
        ))

    return out


def _find_value_for(label_idx: int, lines: list[TextLine]) -> Optional[int]:
    """Find the nearest line to the RIGHT or BELOW the label as the value."""
    label_center = _bbox_center(lines[label_idx].bbox)
    if label_center is None:
        return None
    lx, ly = label_center

    best_idx      = None
    best_distance = float("inf")
    for i, line in enumerate(lines):
        if i == label_idx:
            continue
        c = _bbox_center(line.bbox)
        if c is None:
            continue
        cx, cy = c
        if cx >= lx and abs(cy - ly) < 40:
            distance = cx - lx
        elif cy >= ly and abs(cx - lx) < 200:
            distance = (cy - ly) * 1.5
        else:
            continue
        if distance < best_distance:
            best_distance = distance
            best_idx      = i

    return best_idx


_ENRICH_MIN_VALUE_LEN = 4   # below this, fuzzy match produces too many false positives
_ENRICH_MIN_SCORE     = 70  # partial_ratio score threshold

# Field keys whose MRZ values use formats that don't match what the document
# prints (dates are YYMMDD in MRZ, free format on the layout). We skip them.
_ENRICH_SKIP_KEYS = {"birth_date", "expiry_date", "sex"}


def enrich_with_ocr_positions(
    suggestions: list[Suggestion],
    lines: list[TextLine],
) -> list[Suggestion]:
    """Best-effort: for suggestions that have a value_preview but no
    value_line_ids, fuzzy-match the value against the OCR text and populate
    the line IDs when a clear match is found.

    Suggestions whose value cannot be located are returned UNCHANGED (with
    empty line_ids) — the consumer can still surface them as text-only fields
    and let the user accept or discard them in the confirm step."""
    if not lines:
        return list(suggestions)

    out: list[Suggestion] = []
    for s in suggestions:
        if s.value_line_ids or s.key in _ENRICH_SKIP_KEYS:
            out.append(s)
            continue

        value = (s.value_preview or "").strip()
        if len(value) < _ENRICH_MIN_VALUE_LEN:
            out.append(s)
            continue

        best_idx, best_score = None, 0
        for i, line in enumerate(lines):
            if not line.text:
                continue
            score = fuzz.partial_ratio(value.lower(), line.text.lower())
            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx is None or best_score < _ENRICH_MIN_SCORE:
            out.append(s)
            continue

        label_idx = s.label_line_id
        if label_idx is None:
            label_idx = _find_label_for(best_idx, lines)

        out.append(Suggestion(
            key            = s.key,
            label          = s.label,
            type           = s.type,
            value_preview  = s.value_preview,
            label_line_id  = label_idx,
            value_line_ids = [best_idx],
            confidence     = s.confidence,
            source         = s.source,
        ))

    return out


def extract_anchors(lines: list[TextLine], image_height: int) -> list[str]:
    """Top-N short header lines: y in top 15% of image, length 5-40 chars, no digits."""
    if not lines or image_height <= 0:
        return []
    threshold = image_height * 0.15
    anchors: list[str] = []
    for line in lines:
        if not (5 <= len(line.text.strip()) <= 40):
            continue
        if _LABEL_HAS_DIGITS.search(line.text):
            continue
        center = _bbox_center(line.bbox)
        if center is None or center[1] >= threshold:
            continue
        anchors.append(line.text.strip())
        if len(anchors) >= 5:
            break
    return anchors
