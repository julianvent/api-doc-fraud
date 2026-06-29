"""HTTP handlers for the template generation flow (no VLM).

  GET    /v1/templates             → list_templates
  GET    /v1/templates/{slug}      → get_template
  POST   /v1/templates/generate    → generate_template (no persistence)
  POST   /v1/templates/confirm     → confirm_template (writes JSON + indexes in Qdrant)

Persistence model: each template is a JSON file on disk under
service/ocr/templates/{document_type}_{country_iso}_{edition}.json
plus a point in Qdrant for vector matching. NO database row.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import HTTPException, UploadFile

from api.v1.schema.document_template import (
    TemplateDetail,
    TemplateField,
    TemplateSummary,
)
from api.v1.schema.template_confirm import ConfirmTemplateRequest
from api.v1.schema.template_generate import (
    BBoxRegion,
    FieldSuggestion,
    GenerateResponse,
    OCRElement,
    OCRLine,
    PreclassPayload,
)
from service.logging_config import get_logger
from service.ocr.engine import load_image
from service.ocr.models import Config as OCRConfig
from service.ocr.ocr import _get_engine
from service.ocr.preclassifier import classify as preclassify
from service.preprocessor import preprocessor
from service.template_ocr import heuristics, scan_cache


log = get_logger(__name__)

TEMPLATES_DIR       = Path(OCRConfig().templates_dir)
TEMPLATE_IMAGES_DIR = Path("template")
TEMPLATES_JSON_DIR  = Path("templates_json")


# Lazy singleton for DotsOCRAdapter — loading the model is expensive.
_dots_engine_instance = None


def _get_dots_engine():
    global _dots_engine_instance
    if _dots_engine_instance is None:
        from service.ocr.engine import DotsOCRAdapter
        _dots_engine_instance = DotsOCRAdapter(OCRConfig())
    return _dots_engine_instance


def _normalize_bbox_to_region(bbox: "np.ndarray", w: int, h: int) -> dict:
    """Convert a 4-point pixel bbox [[x,y],...] to a normalised {x1,y1,x2,y2} dict."""
    xs = bbox[:, 0]
    ys = bbox[:, 1]
    return {
        "x1": float(xs.min() / w),
        "y1": float(ys.min() / h),
        "x2": float(xs.max() / w),
        "y2": float(ys.max() / h),
    }


# ─────────────────────────────────────────── filename helpers


_SLUG_RE = re.compile(r"^(?P<document_type>[^/]+?)_(?P<country>[A-Z]{3}|any)_(?P<edition>\d{1,5})$")


def _template_slug(document_type: str, country_iso: Optional[str], edition: int) -> str:
    iso = (country_iso or "any").upper()
    return f"{document_type}_{iso}_{edition}"


def _template_path(slug: str) -> Path:
    return TEMPLATES_DIR / f"{slug}.json"


def _parse_slug(slug: str) -> Optional[tuple[str, Optional[str], int]]:
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    iso = m.group("country")
    return (
        m.group("document_type"),
        None if iso == "ANY" else iso,
        int(m.group("edition")),
    )


# ─────────────────────────────────────────── list / get


def list_templates(
    document_type: Optional[str] = None,
    country: Optional[str] = None,
) -> list[TemplateSummary]:
    if not TEMPLATES_DIR.exists():
        return []

    out: list[TemplateSummary] = []
    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if document_type and data.get("document_type") != document_type:
            continue
        if country and data.get("country") != country:
            continue
        out.append(_data_to_summary(path.stem, data))
    return out


def get_template(template_id: str) -> TemplateDetail:
    path = _template_path(template_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="template not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"template file unreadable: {e}") from e
    return _data_to_detail(template_id, data)


# ─────────────────────────────────────────── generate


def generate_template(
    image: UploadFile,
    mode: str,
) -> GenerateResponse:
    if mode not in {"auto", "manual", "dots"}:
        raise HTTPException(status_code=422, detail="mode must be 'auto', 'manual', or 'dots'")
    from controller._upload_limits import read_within_limit

    image_bytes = read_within_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="empty image upload")

    ext = (Path(image.filename or "").suffix.lstrip(".") or "jpg").lower()
    generate_id = scan_cache.save(image_bytes, extension=ext)
    cached_path = scan_cache.path_for(generate_id)
    if cached_path is None:
        raise HTTPException(status_code=500, detail="failed to persist upload to scan cache")

    # Reuse the same preprocessing pipeline the verify path uses. This handles
    # PDFs (rasterized via fitz), deskew, contrast, etc. — the template OCR
    # quality is what determines downstream verify quality, so we want the
    # same pixels going through both flows.
    try:
        processed_pages = preprocessor.process([cached_path])
    except Exception as e:
        log.error("preprocessor failed on %s: %s: %s", cached_path.name, type(e).__name__, e)
        raise HTTPException(
            status_code=422,
            detail=f"could not preprocess upload (unsupported or corrupt file?): {e}",
        ) from e

    if not processed_pages:
        raise HTTPException(status_code=422, detail="preprocessor returned no pages")

    np_img   = _ensure_rgb(processed_pages[0].image)
    h, width = np_img.shape[:2]

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=scan_cache.get_ttl_seconds())

    # ── dots mode: DotsOCR returns all text elements with spatial coordinates.
    # The client receives the raw element list and assigns label/value roles
    # in the UI; no automatic field pairing is performed here.
    if mode == "dots":
        try:
            dots_engine = _get_dots_engine()
        except Exception as e:
            log.error("DotsOCR init failed: %s: %s", type(e).__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"DotsOCR engine unavailable: {e}",
            ) from e

        lines = dots_engine.extract(np_img)

        from service.template_ocr.element_classifier import classify_elements

        elements_payload = classify_elements([
            {
                "id":       i,
                "text":     line.text,
                "category": getattr(line, "category", None) or "Text",
                "bbox":     _normalize_bbox_to_region(line.bbox, width, h),
            }
            for i, line in enumerate(lines)
        ])

        scan_cache.save_elements(generate_id, elements_payload)
        scan_cache.save_preprocessed_image(generate_id, np_img)

        ocr_elements = [
            OCRElement(
                id       = e["id"],
                text     = e["text"],
                category = e["category"],
                bbox     = BBoxRegion(**e["bbox"]),
                role     = e.get("role", "unknown"),
            )
            for e in elements_payload
        ]

        preclass = preclassify(np_img, lines)

        return GenerateResponse(
            generate_id        = generate_id,
            expires_at         = expires_at,
            image_dims         = (int(width), int(h)),
            preclass           = PreclassPayload(
                doc_family  = preclass.doc_family,
                country_iso = preclass.country_iso,
                mrz_type    = preclass.mrz_type,
                confidence  = preclass.confidence,
            ),
            qr_config          = _detect_qr(np_img),
            ocr_elements       = ocr_elements,
        )

    # ── auto / manual mode (PaddleOCR). manual returns neutral detected
    # elements with no suggestions; auto adds heuristic suggestions on top.
    config = OCRConfig()
    engine = _get_engine(config)
    lines  = engine.extract(np_img)

    # Step 2: persist element list so confirm can resolve the same IDs.
    from service.template_ocr.elements import textlines_to_elements
    elements = textlines_to_elements(lines)
    scan_cache.save_elements(generate_id, [e.to_dict() for e in elements])

    # ── manual mode: neutral DetectedElements, no suggestions, no pairing.
    if mode == "manual":
        preclass = preclassify(np_img, lines)
        ocr_lines = [
            OCRLine(
                id         = e.id,
                text       = e.text,
                bbox       = e.bbox.tolist(),
                confidence = float(e.confidence),
            )
            for e in elements
        ]
        return GenerateResponse(
            generate_id        = generate_id,
            expires_at         = expires_at,
            image_dims         = (int(width), int(h)),
            preclass           = PreclassPayload(
                doc_family  = preclass.doc_family,
                country_iso = preclass.country_iso,
                mrz_type    = preclass.mrz_type,
                confidence  = preclass.confidence,
            ),
            qr_config          = _detect_qr(np_img),
            ocr_lines          = ocr_lines,
        )

    preclass = preclassify(np_img, lines)

    from service.ocr.mrz import detect as detect_mrz
    mrz_result = detect_mrz(lines)
    mrz_fields = _mrz_to_dict(mrz_result) if mrz_result else None

    suggestions = []
    # mode is always "auto" here; "manual" already returned above.
    suggestions.extend(heuristics.suggest_from_mrz(mrz_result))
    suggestions.extend(heuristics.suggest_from_regex(lines))

    # Best-effort: populate value_element_ids / label_element_id for MRZ-derived
    # suggestions so the client has the element IDs if it wants to render
    # an overlay or jump to the location at confirm time. (Step 7: wire real IDs.)
    suggestions = heuristics.enrich_with_ocr_positions(suggestions, lines)
    suggestions = _dedupe_by_key(suggestions)
    anchors     = heuristics.extract_anchors(lines, image_height=np_img.shape[0])

    ocr_lines = [
        OCRLine(
            id         = i,
            text       = line.text,
            bbox       = [[float(p[0]), float(p[1])] for p in line.bbox],
            confidence = float(line.confidence),
        )
        for i, line in enumerate(lines)
    ]

    return GenerateResponse(
        generate_id        = generate_id,
        expires_at         = expires_at,
        image_dims         = (int(width), int(h)),
        preclass           = PreclassPayload(
            doc_family  = preclass.doc_family,
            country_iso = preclass.country_iso,
            mrz_type    = preclass.mrz_type,
            confidence  = preclass.confidence,
        ),
        qr_config          = _detect_qr(np_img),
        ocr_lines          = ocr_lines,
        mrz_fields         = mrz_fields,
        suggestions        = [FieldSuggestion(**s.to_dict()) for s in suggestions],
        anchors_candidates = anchors,
    )


# ─────────────────────────────────────────── confirm


def confirm_template(req: ConfirmTemplateRequest) -> TemplateDetail:
    slug = _template_slug(req.document_type, req.country_iso, req.edition)
    path = _template_path(slug)

    if path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"template already exists for "
                f"(document_type={req.document_type}, country_iso={req.country_iso}, "
                f"edition={req.edition}) at {path.name}"
            ),
        )

    # Load auxiliary dots-mode data first (before deleting the scan cache entry).
    elements_by_id: dict[int, dict] = {}
    ext = "jpg"
    if req.generate_id:
        cached_elements = scan_cache.load_elements(req.generate_id)
        if cached_elements:
            elements_by_id = {e["id"]: e for e in cached_elements}

    img_path: Optional[str] = None
    if req.generate_id:
        cached = scan_cache.load(req.generate_id)
        if cached is None:
            raise HTTPException(
                status_code=410,
                detail="generate_id expired or unknown; re-upload the image",
            )
        ext      = scan_cache.extension_for(req.generate_id) or "jpg"
        img_path = _persist_template_image(cached, slug, ext)
        scan_cache.delete(req.generate_id)

    # Build the fields list, resolving element IDs to spatial regions where
    # provided. label_element_id / value_element_id are not persisted.
    _coord_fields = {"label_element_id", "value_element_id"}
    fields_payload = []
    for f in req.fields:
        fd = f.model_dump(exclude=_coord_fields)
        if f.label_element_id is not None:
            elem = elements_by_id.get(f.label_element_id)
            if elem:
                fd["label_region"] = elem["bbox"]
        if f.value_element_id is not None:
            elem = elements_by_id.get(f.value_element_id)
            if elem:
                fd["value_region"] = elem["bbox"]
        fields_payload.append(fd)

    data = {
        "schema_version": 2,
        "document_type":  req.document_type,
        "document_name":  req.document_name,
        "country":        req.country,
        "country_iso":    req.country_iso,
        "state":          req.state,
        "edition":        req.edition,
        "doc_family":     req.doc_family,
        "mrz_type":       req.mrz_type,
        "img_path":       img_path,
        "fields":         fields_payload,
        "anchors":        list(req.anchors or []),
        "fingerprint":    req.fingerprint or {},
        "field_rules":    req.field_rules or {},
        "qr_config":      req.qr_config or {},
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write to the per-template folder (enriched copy with coordinates).
    _persist_templates_json(slug, ext, data, req.generate_id)

    # Clean up any remaining dots-mode cache files.
    if req.generate_id:
        scan_cache.delete_dots_cache(req.generate_id)

    _index_template_in_qdrant(slug, data)

    return _data_to_detail(slug, data)


# ─────────────────────────────────────────── helpers


def _persist_templates_json(
    slug: str,
    ext: str,
    data: dict,
    generate_id: Optional[str],
) -> None:
    """Write enriched template JSON + preprocessed image to templates_json/{slug}/."""
    import shutil

    tj_dir = TEMPLATES_JSON_DIR / slug
    tj_dir.mkdir(parents=True, exist_ok=True)
    (tj_dir / "template.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if generate_id:
        preprocessed = scan_cache.path_for_preprocessed(generate_id)
        if preprocessed and preprocessed.exists():
            shutil.copy2(preprocessed, tj_dir / f"preprocessed.{ext}")


def _persist_template_image(image_bytes: bytes, slug: str, ext: str) -> str:
    dest_dir = TEMPLATE_IMAGES_DIR / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"template.{ext}"
    dest.write_bytes(image_bytes)
    return dest.as_posix()


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """PaddleOCR (and most other downstream consumers) require H×W×3 RGB.
    The preprocessor may return grayscale (H×W), grayscale-with-alpha (H×W×2),
    BGRA (H×W×4), or RGBA. Normalize to RGB here."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return np.repeat(image, 3, axis=2)
        if channels == 3:
            return image
        if channels == 4:
            return image[:, :, :3]
    raise ValueError(f"unsupported image shape {image.shape}")


def _detect_qr(image: np.ndarray) -> dict:
    try:
        import cv2
        detector       = cv2.QRCodeDetector()
        bgr            = image[:, :, ::-1] if image.ndim == 3 else image
        data, _, _     = detector.detectAndDecode(bgr)
        if not data:
            return {"present": False, "signed": False}
        return {"present": True, "signed": False, "format": "qr"}
    except Exception:
        return {"present": False, "signed": False}


def _mrz_to_dict(mrz) -> dict:
    return {
        "surname":         mrz.surname,
        "given_names":     mrz.given_names,
        "country":         mrz.country,
        "birth_date":      mrz.birth_date,
        "expiry_date":     mrz.expiry_date,
        "document_number": mrz.number,
        "sex":             mrz.sex,
    }


def _dedupe_by_key(suggestions: list) -> list:
    out  = []
    seen = set()
    for s in sorted(suggestions, key=lambda x: 0 if x.confidence == "high" else 1):
        if s.key in seen:
            continue
        seen.add(s.key)
        out.append(s)
    return out


def _data_to_summary(slug: str, data: dict) -> TemplateSummary:
    return TemplateSummary(
        id            = slug,
        document_type = data.get("document_type", ""),
        country       = data.get("country"),
        edition       = int(data.get("edition") or 0),
        document_name = data.get("document_name", ""),
        created_at    = _parse_dt(data.get("created_at")),
    )


def _data_to_detail(slug: str, data: dict) -> TemplateDetail:
    fields = [TemplateField(**f) for f in (data.get("fields") or []) if isinstance(f, dict)]
    return TemplateDetail(
        id              = slug,
        schema_version  = int(data.get("schema_version") or 2),
        document_type   = data.get("document_type", ""),
        document_name   = data.get("document_name", ""),
        country         = data.get("country"),
        country_iso     = data.get("country_iso"),
        state           = data.get("state"),
        edition         = int(data.get("edition") or 0),
        doc_family      = data.get("doc_family"),
        mrz_type        = data.get("mrz_type"),
        img_path        = data.get("img_path"),
        fields          = fields,
        anchors         = list(data.get("anchors") or []),
        fingerprint     = data.get("fingerprint") or {},
        field_rules     = data.get("field_rules") or {},
        qr_config       = data.get("qr_config") or {},
        created_at      = _parse_dt(data.get("created_at")),
    )


def _parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _index_template_in_qdrant(slug: str, data: dict) -> None:
    try:
        from service.ocr import matching
        from service.ocr.preclassifier import PreClassResult
        from service.template_ocr.schema import Fingerprint
    except Exception as e:
        log.warning("qdrant deps unavailable: %s", e)
        return

    cfg = OCRConfig()
    if cfg.disable_vector_match or not matching.is_available():
        log.info("qdrant disabled / unavailable — skipping upsert for slug=%s", slug)
        return

    fp_data = data.get("fingerprint") or {}
    fp = Fingerprint(
        layout_desc = fp_data.get("layout_desc"),
        anchors     = list(data.get("anchors") or []),
    )
    preclass = PreClassResult(
        doc_family  = data.get("doc_family") or "unknown",
        country_iso = data.get("country_iso"),
        mrz_type    = data.get("mrz_type"),
    )
    text = matching.serialize_for_template(fp, preclass)
    if not text:
        log.warning("empty fingerprint text — skipping qdrant upsert for slug=%s", slug)
        return

    vector = matching.embed(text, cfg.embedding_url, cfg.embedding_model)
    if vector is None:
        return

    payload = {
        "template_id":   slug,
        "document_type": data.get("document_type"),
        "country":       data.get("country"),
        "country_iso":   data.get("country_iso"),
        "edition":       data.get("edition"),
        "doc_family":    data.get("doc_family"),
        "mrz_type":      data.get("mrz_type"),
    }
    matching.upsert(
        cfg.qdrant_url,
        cfg.qdrant_collection,
        point_id    = slug,
        vector      = vector,
        payload     = payload,
        vector_size = len(vector),
    )
