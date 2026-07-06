import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image as PILImage
from rapidfuzz import fuzz

from .models import TextLine


TEMPLATES_DIR         = Path(__file__).parent.parent / "template_ocr" / "templates"
BBOX_TOLERANCE_X = 0.04   # horizontal — minor x-drift after alignment
BBOX_TOLERANCE_Y = 0.01   # vertical   — tight to avoid capturing adjacent rows
MATCH_THRESHOLD       = 0.70
LABEL_FUZZY_THRESHOLD = 70

_MIN_GOOD_MATCHES = 10


def _load_ref_image(template: dict) -> np.ndarray | None:
    """Return the _preprocessed reference image for a template as an RGB numpy array.

    Resolution order:
    1. reference_image field, resolved relative to the template's own directory.
    2. Any *_preprocessed.png file found in the template directory (auto-detect).
    """
    template_dir = template.get("_template_dir")

    explicit = template.get("reference_image")
    if explicit and template_dir:
        path = Path(template_dir) / explicit
        if path.exists():
            return np.array(PILImage.open(path).convert("RGB"))

    if template_dir:
        for candidate in sorted(Path(template_dir).glob("*_preprocessed.png")):
            return np.array(PILImage.open(candidate).convert("RGB"))

    return None


def align_to_template(
    doc_image: np.ndarray,
    template: dict,
) -> tuple[np.ndarray, bool]:
    """Warp doc_image to match the template reference image using ORB + partial affine.

    Uses estimateAffinePartial2D (translation + rotation + uniform scale, 4 DoF)
    instead of findHomography (8 DoF) to avoid perspective distortion on flat documents.
    Returns (aligned_image, True) on success, (doc_image, False) otherwise.
    """
    ref_image = _load_ref_image(template)
    if ref_image is None:
        return doc_image, False

    gray_doc = cv2.cvtColor(doc_image, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(ref_image,  cv2.COLOR_RGB2GRAY)

    orb             = cv2.ORB_create(nfeatures=3000)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_doc, des_doc = orb.detectAndCompute(gray_doc, None)

    if des_ref is None or des_doc is None:
        return doc_image, False

    # knnMatch + Lowe's ratio test — eliminates ambiguous matches that cause
    # distortion when passed to the transform estimator.
    bf         = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw        = bf.knnMatch(des_ref, des_doc, k=2)
    good       = [m for pair in raw if len(pair) == 2
                  for m, n in [pair] if m.distance < 0.75 * n.distance]
    good       = sorted(good, key=lambda m: m.distance)[:200]

    if len(good) < _MIN_GOOD_MATCHES:
        print(f"[ALIGN] not enough good matches: {len(good)} < {_MIN_GOOD_MATCHES}")
        return doc_image, False

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_doc[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Partial affine: translation + rotation + uniform scale only.
    # Avoids the shear / perspective deformation that full homography allows.
    M, mask = cv2.estimateAffinePartial2D(
        dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    inliers = int(mask.sum()) if mask is not None else 0

    if M is None or inliers < _MIN_GOOD_MATCHES:
        print(f"[ALIGN] affine estimation failed: inliers={inliers}")
        return doc_image, False

    # Sanity checks — reject transforms that are physically implausible.
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    angle = float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))
    if not (0.5 <= scale <= 2.0):
        print(f"[ALIGN] implausible scale {scale:.2f} — skipping alignment")
        return doc_image, False
    if abs(angle) > 30:
        print(f"[ALIGN] implausible rotation {angle:.1f}° — skipping alignment")
        return doc_image, False

    h, w    = ref_image.shape[:2]
    aligned = cv2.warpAffine(doc_image, M, (w, h), flags=cv2.INTER_LINEAR)
    print(f"[ALIGN] ok — inliers={inliers}/{len(good)}, scale={scale:.3f}, angle={angle:.1f}°")
    return aligned, True


@dataclass
class MatchResult:
    matched          : bool
    match_score      : float
    document_type    : str
    document_name    : str
    field_lines      : dict[str, list[TextLine]]
    unmatched_fields : list[str]


def _template_year(data: dict) -> int | None:
    """Extract the template edition year from JSON fields or directory name."""
    for key in ("edition", "year"):
        if key in data:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    import re
    template_dir = data.get("_template_dir", "")
    m = re.search(r'_(\d{4})(?:[_/\\]|$)', template_dir)
    return int(m.group(1)) if m else None


def _load_template_from_dir(subdir: Path) -> dict | None:
    """Load and enrich a template JSON from its subdirectory.
    Injects '_template_dir' so image resolution works relative to the folder."""
    json_files = list(subdir.glob("*.json"))
    if not json_files:
        return None
    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_template_dir"] = str(subdir)
        return data
    except Exception:
        return None


def load_template(document_type: str) -> dict | None:
    """Return the most recent template for the given document_type, or None."""
    templates = load_templates(document_type)
    return templates[0] if templates else None


def load_templates(document_type: str, issue_year: int | None = None) -> list[dict]:
    """Load all templates for document_type from TEMPLATES_DIR subdirectories.
    Each template lives in its own folder: TEMPLATES_DIR/{type}_{country}_{year}/.
    If issue_year is given, returns only editions <= issue_year, most recent first.
    Without issue_year, returns all editions most recent first."""
    with_year: list[tuple[int | None, dict]] = []

    if not TEMPLATES_DIR.exists():
        return []

    for subdir in TEMPLATES_DIR.iterdir():
        if not subdir.is_dir():
            continue
        data = _load_template_from_dir(subdir)
        if data is None:
            continue
        if data.get("document_type") != document_type:
            continue
        with_year.append((_template_year(data), data))

    if issue_year is not None:
        eligible = [(y, t) for y, t in with_year if y is None or y <= issue_year]
        print(
            f"[TMPL] issue_year={issue_year} → eligible: "
            f"{[(y, t.get('document_name', '?')) for y, t in eligible]}"
        )
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


def _overlap_ratio(
    line: TextLine, region: dict,
    tol_x: float = BBOX_TOLERANCE_X,
    tol_y: float = BBOX_TOLERANCE_Y,
) -> float:
    """Fraction of the line's bbox area that falls inside region (expanded by tol_x/tol_y)."""
    bounds = _bbox_bounds(line)
    if bounds is None:
        return 0.0
    bx1, by1, bx2, by2 = bounds
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return 0.0
    rx1, ry1 = region["x1"] - tol_x, region["y1"] - tol_y
    rx2, ry2 = region["x2"] + tol_x, region["y2"] + tol_y
    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / bbox_area


def _line_in_region(
    line: TextLine, region: dict,
    tol_x: float = BBOX_TOLERANCE_X,
    tol_y: float = BBOX_TOLERANCE_Y,
) -> bool:
    return _overlap_ratio(line, region, tol_x, tol_y) >= OVERLAP_THRESHOLD


def _find_lines_in_region(
    lines: list[TextLine], region: dict,
    tol_x: float = BBOX_TOLERANCE_X,
    tol_y: float = BBOX_TOLERANCE_Y,
) -> list[TextLine]:
    return [l for l in lines if _line_in_region(l, region, tol_x, tol_y)]


def _cy(line: TextLine) -> float:
    ys = [pt[1] for pt in line.bbox]
    return (min(ys) + max(ys)) / 2



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

    for field_def in fields:
        key            = field_def.get("key", "")
        expected_label = field_def.get("label", "")
        label_region   = field_def.get("label_region")
        value_region   = field_def.get("value_region")
        label_latin    = _latin_part(expected_label)

        # Positional-only field: no visible label to search for.
        # Detected when label_region == value_region (confirm mirrored them because
        # the user selected the same elements for both, e.g. MRZ, visa number).
        # Also applies when label text is empty after stripping non-Latin chars.
        _positional = not label_latin or (
            label_region is not None and label_region == value_region
        )

        if _positional:
            # Count as matched if the value region actually contains OCR text.
            value_candidates = _find_lines_in_region(lines, value_region) if value_region else []
            label_found      = bool(value_candidates)
            label_line       = None
        elif label_region:
            label_candidates = _find_lines_in_region(lines, label_region)
            label_line       = _find_label_line(label_candidates, label_latin)
            label_found      = label_line is not None
        else:
            # fallback: global text search for templates without label_region
            ocr_text    = " ".join(l.text for l in lines)
            label_found = fuzz.partial_ratio(label_latin, ocr_text) >= LABEL_FUZZY_THRESHOLD
            label_line  = _find_label_line(lines, label_latin) if label_found else None

        if label_found:
            field_hits += 1
            value_lines = _find_lines_in_region(lines, value_region) if value_region else []

            if _positional:
                # Return all lines in the region (e.g. multi-row MRZ, visa number).
                # No truncation — the value spans as many rows as the region contains.
                pass
            else:
                # Exclude only the matched label line for THIS field.
                # A global exclude-list would incorrectly remove lines that are
                # valid values for neighbouring fields with overlapping regions.
                if label_line is not None:
                    value_lines = [l for l in value_lines if l is not label_line]
                # All remaining lines in the bbox are part of the value —
                # _row_order in ocr.py will sort and join them.

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
