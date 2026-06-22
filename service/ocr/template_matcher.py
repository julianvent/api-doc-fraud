import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image as PILImage
from rapidfuzz import fuzz

from .models import TextLine


TEMPLATES_DIR         = Path(__file__).parent / "templates"
BBOX_TOLERANCE        = 0.04
MATCH_THRESHOLD       = 0.80
LABEL_FUZZY_THRESHOLD = 70

_REF_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
_MIN_GOOD_MATCHES = 10


def _load_ref_image(template: dict) -> np.ndarray | None:
    """Return the reference image for a template as an RGB numpy array, or None."""
    explicit = template.get("reference_image")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = TEMPLATES_DIR / explicit
        if path.exists():
            return np.array(PILImage.open(path).convert("RGB"))
        return None

    doc_type = template.get("document_type", "")
    for ext in _REF_EXTENSIONS:
        path = TEMPLATES_DIR / f"{doc_type}{ext}"
        if path.exists():
            return np.array(PILImage.open(path).convert("RGB"))
    return None


def align_to_template(
    doc_image: np.ndarray,
    template: dict,
) -> tuple[np.ndarray, bool]:
    """Warp doc_image to match the template reference image using ORB + homography.

    Returns (aligned_image, True) on success, (doc_image, False) if the reference
    image is missing or not enough keypoints are matched.
    """
    ref_image = _load_ref_image(template)
    if ref_image is None:
        return doc_image, False

    gray_doc = cv2.cvtColor(doc_image, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(ref_image,  cv2.COLOR_RGB2GRAY)

    orb              = cv2.ORB_create(nfeatures=3000)
    kp_ref, des_ref  = orb.detectAndCompute(gray_ref, None)
    kp_doc, des_doc  = orb.detectAndCompute(gray_doc, None)

    if des_ref is None or des_doc is None:
        return doc_image, False

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_ref, des_doc), key=lambda m: m.distance)
    good    = matches[: max(_MIN_GOOD_MATCHES, len(matches) // 3)]

    if len(good) < _MIN_GOOD_MATCHES:
        return doc_image, False

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_doc[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None or int(mask.sum()) < _MIN_GOOD_MATCHES:
        return doc_image, False

    h, w    = ref_image.shape[:2]
    aligned = cv2.warpPerspective(doc_image, H, (w, h), flags=cv2.INTER_LINEAR)
    return aligned, True


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


def _template_year(data: dict, path: Path) -> int | None:
    """Extract template year: first from the 'year' JSON field, then from the filename."""
    if "year" in data:
        try:
            return int(data["year"])
        except (ValueError, TypeError):
            pass
    import re
    m = re.search(r'_(\d{4})$', path.stem)
    return int(m.group(1)) if m else None


def load_templates(document_type: str, issue_year: int | None = None) -> list[dict]:
    """Load templates of the given type. If issue_year is provided, returns only
    templates with year <= issue_year, sorted most recent first.
    Without issue_year, returns all templates sorted most recent first."""
    candidates = sorted(TEMPLATES_DIR.glob(f"{document_type}*.json"))
    with_year: list[tuple[int | None, dict]] = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            year = _template_year(data, path)
            with_year.append((year, data))
        except Exception:
            continue

    if issue_year is not None:
        eligible = [(y, t) for y, t in with_year if y is None or y <= issue_year]
        print(f"[TMPL] issue_year={issue_year} → eligible templates: "
              f"{[(y, t.get('document_name', '?')) for y, t in eligible]}")
        if eligible:
            eligible.sort(key=lambda x: x[0] or 0, reverse=True)
            return [t for _, t in eligible]

    with_year.sort(key=lambda x: x[0] or 0, reverse=True)
    return [t for _, t in with_year]


def _line_center(line: TextLine) -> tuple[float, float] | None:
    if line.bbox is None or len(line.bbox) == 0:
        return None
    xs = [pt[0] for pt in line.bbox]
    ys = [pt[1] for pt in line.bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


OVERLAP_THRESHOLD = 0.5


def _bbox_bounds(line: TextLine) -> tuple[float, float, float, float] | None:
    if line.bbox is None or len(line.bbox) == 0:
        return None
    xs = [pt[0] for pt in line.bbox]
    ys = [pt[1] for pt in line.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_ratio(line: TextLine, region: dict, tolerance: float = BBOX_TOLERANCE) -> float:
    """Fraction of the bbox area that falls inside region (expanded by tolerance)."""
    bounds = _bbox_bounds(line)
    if bounds is None:
        return 0.0
    bx1, by1, bx2, by2 = bounds
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return 0.0

    rx1 = region["x1"] - tolerance
    ry1 = region["y1"] - tolerance
    rx2 = region["x2"] + tolerance
    ry2 = region["y2"] + tolerance

    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return ((ix2 - ix1) * (iy2 - iy1)) / bbox_area


def _line_in_region(line: TextLine, region: dict, tolerance: float = BBOX_TOLERANCE) -> bool:
    return _overlap_ratio(line, region, tolerance) >= OVERLAP_THRESHOLD


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

    field_lines      : dict[str, list[TextLine]] = {}
    unmatched_fields : list[str]                 = []
    field_hits       = 0

    # Collect all label-region lines upfront to exclude them from value extraction
    all_label_ids: set[int] = set()
    for fd in fields:
        lr = fd.get("label_region")
        if lr:
            all_label_ids.update(id(l) for l in _find_lines_in_region(lines, lr, tolerance=0.0))

    for field_def in fields:
        key            = field_def.get("key", "")
        expected_label = field_def.get("label", "")
        label_region   = field_def.get("label_region")
        value_region   = field_def.get("value_region")
        label_latin    = _latin_part(expected_label)

        # ── positional label match ─────────────────────────────────────────
        if label_region:
            label_candidates = _find_lines_in_region(lines, label_region)
            label_line       = _find_label_line(label_candidates, label_latin) if label_latin else None
            label_found      = label_line is not None
        else:
            # fallback: global text search for templates without label_region
            ocr_text    = " ".join(l.text for l in lines)
            label_found = fuzz.partial_ratio(label_latin, ocr_text) >= LABEL_FUZZY_THRESHOLD if label_latin else False
            label_line  = _find_label_line(lines, label_latin) if label_found else None

        if label_found:
            field_hits  += 1
            value_lines  = _find_lines_in_region(lines, value_region) if value_region else []
            value_lines  = [l for l in value_lines if id(l) not in all_label_ids]

            if label_line is not None:
                value_lines = [l for l in value_lines if _cy(l) > _cy(label_line)]

            value_lines      = _first_row(value_lines)
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
