import logging
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from api.v1.schema.verify import Identity
from service.ocr.backends.ollama_vision import OllamaVisionBackend
from service.ocr.extractor import extract_with_vision
from service.ocr.template_matcher import (
    align_to_template,
    load_templates,
    match as match_template,
)
from service.ocr.visualizer import visualize_matplotlib, visualize_template_match

from .engine import (
    DolphinOCRAdapter,
    DotsOCRAdapter,
    OCREngine,
    PaddleOCRAdapter,
    load_image,
)
from .models import Config, PipelineOutput
from .mrz import detect, mrz_to_dict as _mrz_to_dict
from .normalizer import normalize_date, normalize_fields
from .preclassifier import classify as preclassify

log = logging.getLogger(__name__)

_ENGINES: dict[str, type[OCREngine]] = {
    "paddle": PaddleOCRAdapter,
    "dots": DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache: dict[str, OCREngine] = {}
_engine_lock = threading.Lock()

_vision_backend: OllamaVisionBackend | None = None
_vision_lock = threading.Lock()

_config: Config | None = None
_config_lock = threading.Lock()

# Fuzzy threshold for MRZ field comparison — allows single-char OCR noise.
_MRZ_FUZZY_THRESHOLD = 100

_ROW_GROUP_THRESHOLD = 0.03

_MRZ_COMPARABLE_FIELDS = {
    "surname",
    "given_names",
    "date_of_birth",
    "expiry_date",
    "document_number",
    "sex",
    "country",
}

_IDENTITY_COMPARABLE_FIELDS = {"full_name", "date_of_birth", "gender"}

_MRZ_TYPE_TO_DOC: dict[str, str] = {"TD3": "passport", "MRV-A": "visa", "MRV-B": "visa"}

_DOC_FAMILY_FALLBACK: dict[str, str] = {
    "proof_of_address": "proof_of_address",
    "identity_card": "identity_card",
    "identity_photo": "identity_photo",
}


# ── singleton accessors ────────────────────────────────────────────────────────

def _get_config() -> Config:
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = Config()
    return _config


def _get_engine(config: Config) -> OCREngine:
    name = config.ocr_engine.lower().strip()
    if name not in _engine_cache:
        with _engine_lock:
            if name not in _engine_cache:
                cls = _ENGINES.get(name)
                if cls is None:
                    raise ValueError(
                        f"Unknown OCR engine: {name!r}. Available: {sorted(_ENGINES)}"
                    )
                _engine_cache[name] = cls(config)
    return _engine_cache[name]


def _get_vision_backend(config: Config) -> OllamaVisionBackend:
    global _vision_backend
    if _vision_backend is None:
        with _vision_lock:
            if _vision_backend is None:
                _vision_backend = OllamaVisionBackend(
                    config.ollama_vision_url, config.ollama_vision_model
                )
    return _vision_backend


def warmup() -> None:
    config = _get_config()
    _get_engine(config)
    _get_vision_backend(config)


# ── helpers ────────────────────────────────────────────────────────────────────

def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def _extract_issue_year(lines: list) -> int | None:
    """Scan probe OCR lines for a date_of_issue candidate.
    Heuristic: collect all non-future dates, drop the oldest (likely birth date),
    take the largest remaining year as the estimated issue year.
    Expiry dates are in the future and are filtered out.
    """
    from datetime import datetime

    current_year = datetime.now().year
    found: list[int] = []
    for line in lines:
        normalized = normalize_date(line.text.strip())
        if not normalized or normalized == line.text.strip():
            continue
        try:
            dt = datetime.strptime(normalized, "%Y/%m/%d")
            if dt.year <= current_year:
                found.append(dt.year)
        except ValueError:
            continue
    if not found:
        return None
    found_sorted = sorted(set(found))
    candidates = found_sorted[1:] if len(found_sorted) > 1 else found_sorted
    year = max(candidates)
    print(f"[PROBE] dates found={found_sorted} → estimated issue_year={year}")
    return year


def _spatial_layout(lines: list) -> str:
    if not lines:
        return ""
    sorted_lines = sorted(
        lines,
        key=lambda l: (
            (min(pt[1] for pt in l.bbox) + max(pt[1] for pt in l.bbox)) / 2,
            min(pt[0] for pt in l.bbox),
        ),
    )
    parts = []
    for l in sorted_lines:
        ys = [pt[1] for pt in l.bbox]
        xs = [pt[0] for pt in l.bbox]
        yc = (min(ys) + max(ys)) / 2
        xc = min(xs)
        parts.append(f"[y={yc:.3f} x={xc:.3f}] {l.text}")
    return "\n".join(parts)


def _row_order(lines: list) -> list:
    if not lines:
        return lines

    def _cy(l):
        ys = [pt[1] for pt in l.bbox]
        return (min(ys) + max(ys)) / 2

    def _cx(l):
        xs = [pt[0] for pt in l.bbox]
        return (min(xs) + max(xs)) / 2

    sorted_by_y = sorted(lines, key=_cy)
    rows = [[sorted_by_y[0]]]
    for line in sorted_by_y[1:]:
        if _cy(line) - _cy(rows[-1][-1]) <= _ROW_GROUP_THRESHOLD:
            rows[-1].append(line)
        else:
            rows.append([line])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=_cx))
    return ordered


def _norm(value: str) -> str:
    return (normalize_date(str(value)) or str(value)).strip().upper().replace(" ", "")


def _resolve_document_type(image, lines: list) -> str | None:
    """Identify document type: preclassifier → MRZ shortcut → doc_family fallback."""
    preclass = preclassify(image, lines)
    print(
        f"[OCR] preclassifier: family={preclass.doc_family} "
        f"mrz_type={preclass.mrz_type} country={preclass.country_iso} "
        f"confidence={preclass.confidence:.2f}"
    )

    if preclass.mrz_type:
        doc_type = _MRZ_TYPE_TO_DOC.get(preclass.mrz_type)
        if doc_type:
            print(f"[OCR] mrz shortcut: {preclass.mrz_type} → {doc_type}")
            return doc_type

    if preclass.doc_family in _DOC_FAMILY_FALLBACK:
        doc_type = _DOC_FAMILY_FALLBACK[preclass.doc_family]
        print(f"[OCR] doc_family fallback: {preclass.doc_family} → {doc_type}")
        return doc_type

    print("[OCR] document type could not be resolved")
    return None


def _build_pipeline_output(
    document_type: str | None, mrz, lines: list, confidence_avg: float
) -> PipelineOutput:
    mrz_verified = mrz if (mrz and mrz.valid) else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None
    source = "ocr+mrz" if mrz_verified else "ocr+mrz?" if mrz_unverified else "ocr"
    return PipelineOutput(
        document_type=document_type or "unknown",
        mrz_verified=mrz_verified,
        mrz_unverified=mrz_unverified,
        lines=lines,
        source=source,
        confidence_avg=round(confidence_avg, 4),
    )


def _save_visualization(
    image, output: PipelineOutput, image_path: Path, config: Config
) -> None:
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_dir = Path(config.ocr_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_matplotlib(
            image_bgr, output, str(output_dir / f"{image_path.stem}_result.png")
        )
    except Exception as e:
        print(f"[VIS] visualization failed (non-critical): {type(e).__name__}: {e}")


def _field_value(template_fields: dict, canonical_key: str) -> str | None:
    """Look up a template field value by canonical English key.
    Handles both exact keys ('surname') and bilingual template keys
    ('apellidos_surname') via suffix match."""
    if canonical_key in template_fields:
        return template_fields[canonical_key]
    suffix = f"_{canonical_key}"
    for k, v in template_fields.items():
        if k.endswith(suffix):
            return v
    return None


def _verify_anchors(
    lines: list, template: dict
) -> tuple[list[str], list[dict]]:
    """Check template anchor texts against OCR lines.
    An anchor is considered found when partial_ratio >= 70 against the full OCR text.
    Flags when zero anchors match OR more than half are missing.
    Returns (flags, details) where details is [{"text": str, "found": bool}, ...]."""
    anchors = template.get("anchors") or []
    if not anchors:
        return [], []

    all_text = " ".join(l.text for l in lines).upper()
    details = [
        {"text": a, "found": fuzz.partial_ratio(a.upper(), all_text) >= 70}
        for a in anchors
    ]
    found_count = sum(1 for d in details if d["found"])
    print(f"[ANCHOR] checked {len(anchors)} anchors, found {found_count}/{len(anchors)}")

    flags: list[str] = []
    if found_count == 0 or found_count < len(anchors) / 2:
        flags.append("anchor_mismatch")
    return flags, details


def _verify_image_regions(
    image: np.ndarray, template: dict
) -> tuple[list[str], list[dict]]:
    """Check if the submitted document contains a face/photo in each template image_region.
    Uses Haar Cascade frontal-face detection; a face center within the region (±5%) counts.
    Returns (flags, details) where details is [{"region": dict, "face_detected": bool}, ...]."""
    regions = template.get("image_regions") or []
    if not regions:
        return [], []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4,
        minSize=(int(w * 0.04), int(h * 0.04)),
        maxSize=(int(w * 0.60), int(h * 0.60)),
    )

    face_centers: list[tuple[float, float]] = []
    if isinstance(faces, np.ndarray) and len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            face_centers.append(((fx + fw / 2) / w, (fy + fh / 2) / h))

    tol = 0.05
    details = []
    for reg in regions:
        in_region = any(
            reg["x1"] - tol <= cx <= reg["x2"] + tol
            and reg["y1"] - tol <= cy <= reg["y2"] + tol
            for cx, cy in face_centers
        )
        details.append({"region": reg, "face_detected": in_region})
        print(f"[IMG-RGN] region={reg} face_detected={in_region}")

    flags: list[str] = []
    if details and not any(d["face_detected"] for d in details):
        flags.append("image_region_mismatch")
    return flags, details


def _compare_mrz(template_fields: dict, mrz) -> list[dict]:
    mrz_dict = _mrz_to_dict(mrz)
    mismatches = []
    for key in _MRZ_COMPARABLE_FIELDS:
        tv = _field_value(template_fields, key)
        mv = mrz_dict.get(key)
        if not tv or not mv:
            print(f"[MRZ-CMP] skip '{key}': tv={tv!r} mv={mv!r}")
            continue
        tv_n, mv_n = _norm(tv), _norm(mv)
        similarity = fuzz.ratio(tv_n, mv_n)
        status = "MISMATCH" if similarity < _MRZ_FUZZY_THRESHOLD else "ok"
        print(f"[MRZ-CMP] {status} '{key}': ocr={tv_n!r} mrz={mv_n!r} sim={similarity}")
        if similarity < _MRZ_FUZZY_THRESHOLD:
            mismatches.append({
                "field": key,
                "document_value": tv,
                "mrz_value": mv,
            })
    return mismatches


def _compare_identity(template_fields: dict, identity: Identity) -> list[dict]:
    identity_dict = identity.model_dump()
    mismatches = []
    for key in _IDENTITY_COMPARABLE_FIELDS:
        template_value = _field_value(template_fields, key)
        identity_value = identity_dict.get(key)
        if not template_value or not identity_value:
            continue
        if _norm(template_value) != _norm(identity_value):
            mismatches.append({
                "field": key,
                "document_value": template_value,
                "packet_value": identity_value,
            })
    return mismatches


# ── pipeline ───────────────────────────────────────────────────────────────────

def _run(
    image_path: str | Path,
    config: Config,
    vision_backend: OllamaVisionBackend,
    identity_packet: Identity,
    document_type: str | None = None,
) -> dict:

    engine = _get_engine(config)
    image_orig = load_image(Path(image_path))

    # ── probe pass: identify document type and issue year ─────────────────────
    lines_probe: list | None = None
    if not document_type:
        lines_probe = engine.extract(image_orig)
        if not lines_probe:
            return {"error": "no text extracted", "source": None}
        ocr_confidence = _avg_confidence(lines_probe)
        if ocr_confidence < config.confidence_threshold:
            return {
                "error": "low confidence — document quality insufficient",
                "confidence_avg": round(ocr_confidence, 4),
                "source": None,
            }
        document_type = _resolve_document_type(image_orig, lines_probe)
        issue_year = _extract_issue_year(lines_probe)
    else:
        issue_year = None

    templates = load_templates(document_type, issue_year) if document_type else []

    # ── alignment: pre-select the best template to warp to ────────────────────
    # Use probe lines to pick the template whose layout best matches the document
    # so that alignment and field extraction always use the same coordinate system.
    image = image_orig
    align_target = templates[0] if templates else None
    if templates and lines_probe and len(templates) > 1:
        align_target = max(
            templates,
            key=lambda t: match_template(lines_probe, t).match_score,
        )

    if align_target:
        image, aligned_ok = align_to_template(image_orig, align_target)
        print(
            f"[OCR] homography alignment: {'ok' if aligned_ok else 'skipped (no ref image or too few keypoints)'}"
        )

    # ── extraction pass: OCR on aligned image ─────────────────────────────────
    lines = engine.extract(image)

    if not lines:
        return {
            "error": "no text extracted",
            "source": None,
            "document_type": document_type,
        }

    ocr_confidence = _avg_confidence(lines)
    print(f"[OCR] avg OCR confidence: {ocr_confidence:.3f}")

    if ocr_confidence < config.confidence_threshold:
        return {
            "error": "low confidence — document quality insufficient",
            "confidence_avg": round(ocr_confidence, 4),
            "source": None,
        }

    mrz = detect(lines)
    print(f"[MRZ] detected={mrz is not None} valid={mrz.valid if mrz else 'N/A'}")
    if mrz:
        print(
            f"[MRZ] raw → surname={mrz.surname!r} given={mrz.given_names!r} "
            f"dob={mrz.date_of_birth!r} expiry={mrz.expiry_date!r} "
            f"num={mrz.document_number!r} country={mrz.country!r}"
        )

    # ── template path (primary: OCR bbox matching) ────────────────────────────
    if document_type and align_target:
        # Re-score against the pre-selected template using the final aligned lines.
        template_match_confidence = match_template(lines, align_target)
        best_template = align_target
        print(
            f"[TMPL] document_type={document_type!r} "
            f"match_score={template_match_confidence.match_score:.3f} "
            f"matched={template_match_confidence.matched}"
        )

        if not template_match_confidence.matched:
            return {
                "verdict": "unverifiable",
                "flags": ["layout_mismatch"],
                "template_confidence": template_match_confidence.match_score,
                "document_type": document_type,
                "template_available": True,
                "message": "document layout does not match expected template",
            }

        template_fields: dict[str, str | None] = {}
        for key, field_lines in template_match_confidence.field_lines.items():
            if not field_lines:
                template_fields[key] = None
                continue
            ordered = _row_order(field_lines)
            template_fields[key] = " ".join(l.text.strip() for l in ordered).strip()
        template_fields = normalize_fields(template_fields)
        print(f"[TMPL] extracted fields: {template_fields}")

        anchor_flags, anchor_details   = _verify_anchors(lines, best_template)
        img_rgn_flags, img_rgn_details = _verify_image_regions(image, best_template)
        mrz_flags: list[str] = anchor_flags + img_rgn_flags
        mrz_mismatches: list[dict] = []

        if mrz:
            if not mrz.valid:
                mrz_flags.append("mrz_checksum_failed")
                print("[MRZ-CMP] MRZ checksum invalid — comparing anyway")
            mrz_mismatches = _compare_mrz(template_fields, mrz)
            print(f"[MRZ-CMP] mismatches found: {len(mrz_mismatches)}")
            if mrz_mismatches:
                mrz_flags.append("mrz_mismatch")
        else:
            print("[MRZ-CMP] skipped: no MRZ detected")

        identity_mismatches = []
        if identity_packet:
            identity_mismatches = _compare_identity(template_fields, identity_packet)
            print(f"[ID-CMP] mismatches found: {len(identity_mismatches)}")
        else:
            print("[ID-CMP] skipped: no identity packet provided")

        output = _build_pipeline_output(document_type, mrz, lines, ocr_confidence)
        _save_visualization(image, output, Path(image_path), config)

        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vis_path = (
                Path(config.ocr_output_dir) / f"{Path(image_path).stem}_template.png"
            )
            visualize_template_match(
                image_bgr, lines, best_template, template_match_confidence, str(vis_path),
                anchor_details=anchor_details,
                image_region_details=img_rgn_details,
            )
        except Exception as e:
            print(f"[VIS] template visualization failed (non-critical): {type(e).__name__}: {e}")

        return {
            "template_match_confidence": template_match_confidence.match_score,
            "ocr_confidence": ocr_confidence,
            "document_type": document_type,
            "document_name": template_match_confidence.document_name,
            "template_available": True,
            "fields": template_fields,
            "mrz": _mrz_to_dict(mrz) if mrz else None,
            "unmatched_fields": template_match_confidence.unmatched_fields,
            "flags": mrz_flags,
            "anchor_verification": anchor_details,
            "image_region_verification": img_rgn_details,
            "mrz_mismatches": mrz_mismatches,
            "identity_mismatches": identity_mismatches,
        }

    # ── fallback path (no template: full VLM extraction) ──────────────────────
    output = _build_pipeline_output(document_type, mrz, lines, ocr_confidence)
    spatial = _spatial_layout(lines)
    _save_visualization(image, output, Path(image_path), config)

    # Write the preprocessed (aligned) image to a temp file so the VLM always
    # receives the same image that OCR processed — not the original which may
    # be a PDF or un-aligned JPEG that the vision model cannot interpret.
    _fd, _tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(_fd)
    try:
        cv2.imwrite(_tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return extract_with_vision(_tmp, vision_backend, output, spatial, None)
    finally:
        try:
            os.unlink(_tmp)
        except OSError:
            pass


# ── public API ─────────────────────────────────────────────────────────────────

def process(
    file_path: str | list,
    identity: Identity,
    document_type: str | None = None,
) -> dict | list[dict]:
    config = _get_config()
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [
            _run(fp, config, vision_backend, identity, document_type)
            for fp in file_path
        ]

    return _run(file_path, config, vision_backend, identity, document_type)
