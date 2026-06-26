from pathlib import Path
from typing import Optional
from datetime import datetime
import cv2
from rapidfuzz import fuzz

from service.logging_config import get_logger
from service.ocr.backends import OllamaVisionBackend
from service.ocr.identifier import identify
from service.ocr.extractor import _mrz_to_dict, extract_with_vision
from service.ocr.normalizer import normalize_fields, normalize_date

from . import matching
from .layout import build_spatial_layout
from service.template_ocr.schema import load_template as _schema_load
from .template_matcher import match as match_template, load_templates, align_to_template

from .models import Config, PipelineOutput, TextLine
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .mrz import detect
from .preclassifier import classify as preclassify, PreClassResult


log = get_logger(__name__)


_ENGINES: dict[str, type[OCREngine]] = {
    "paddle":  PaddleOCRAdapter,
    "dots":    DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache   : dict[str, OCREngine]       = {}
_engine         : OCREngine | None           = None
_vision_backend : OllamaVisionBackend | None = None


def _get_engine(config: Config) -> OCREngine:
    global _engine
    name = config.ocr_engine.lower().strip()
    if name not in _engine_cache:
        cls = _ENGINES.get(name)
        if cls is None:
            raise ValueError(f"Unknown OCR engine: {name!r}. Available: {sorted(_ENGINES)}")
        _engine_cache[name] = cls(config)
    _engine = _engine_cache[name]
    return _engine


def _get_vision_backend(config: Config) -> OllamaVisionBackend:
    global _vision_backend
    if _vision_backend is None:
        _vision_backend = OllamaVisionBackend(config.ollama_vision_url, config.ollama_vision_model)
    return _vision_backend


def warmup() -> None:
    config = Config()
    _get_engine(config)
    _get_vision_backend(config)


def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def _extract_issue_year(lines: list) -> int | None:
    current_year = datetime.now().year
    found: list[int] = []
    for line in lines:
        normalized = normalize_date(line.text.strip())
        if not normalized or normalized == line.text.strip():
            continue
        try:
            dt = datetime.strptime(normalized, "%d/%m/%Y")
            if dt.year <= current_year:
                found.append(dt.year)
        except ValueError:
            continue
    if not found:
        return None
    found_sorted = sorted(set(found))
    candidates = found_sorted[1:] if len(found_sorted) > 1 else found_sorted
    return max(candidates)


_ROW_GROUP_THRESHOLD = 0.03


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


MRZ_MATCH_THRESHOLD = 90

_MRZ_COMPARABLE_FIELDS = {
    "surname", "given_names", "date_of_birth", "expiry_date",
    "document_number", "sex", "country",
}


def _norm(value: str) -> str:
    return (normalize_date(str(value)) or str(value)).strip().upper().replace(" ", "")


def _compare_mrz(template_fields: dict, mrz) -> list[dict]:
    mrz_dict   = _mrz_to_dict(mrz)
    mismatches = []
    common     = set(template_fields) & set(mrz_dict) & _MRZ_COMPARABLE_FIELDS
    for key in common:
        tv = template_fields.get(key)
        mv = mrz_dict.get(key)
        if not tv or not mv:
            continue
        tv_n, mv_n = _norm(tv), _norm(mv)
        similarity = fuzz.ratio(tv_n, mv_n)
        if similarity < MRZ_MATCH_THRESHOLD:
            mismatches.append({
                "field"         : key,
                "template_value": tv,
                "mrz_value"     : mv,
                "similarity"    : similarity,
            })
    return mismatches


def _build_pipeline_output(document_type, lines, mrz, confidence_avg) -> PipelineOutput:
    mrz_verified   = mrz if (mrz and mrz.valid)     else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None
    source = (
        "ocr+mrz"  if mrz_verified   else
        "ocr+mrz?" if mrz_unverified else
        "ocr"
    )
    return PipelineOutput(
        document_type  = document_type or "unknown",
        mrz_verified   = mrz_verified,
        mrz_unverified = mrz_unverified,
        lines          = lines,
        source         = source,
        confidence_avg = round(confidence_avg, 4),
    )


def _document_type_from_preclass(preclass: PreClassResult) -> str | None:
    if preclass.doc_family != "identity_mrz":
        return None
    mapping = {"TD3": "passport", "MRV-A": "visa", "MRV-B": "visa"}
    return mapping.get(preclass.mrz_type)


def _try_vector_match(
    preclass: PreClassResult,
    lines: list[TextLine],
    config: Config,
) -> Optional[dict]:
    if config.disable_vector_match or not matching.is_available():
        return None

    query_text = matching.serialize_for_query(preclass, lines)
    vector     = matching.embed(query_text, config.embedding_url, config.embedding_model)
    if vector is None:
        return None

    filters = {
        "doc_family":  preclass.doc_family if preclass.doc_family != "unknown" else None,
        "mrz_type":    preclass.mrz_type,
        "country_iso": preclass.country_iso,
    }
    hits = matching.search(
        config.qdrant_url,
        config.qdrant_collection,
        vector,
        filters         = filters,
        limit           = 1,
        score_threshold = config.match_threshold,
    )
    if not hits:
        log.info("qdrant: no match above threshold=%s", config.match_threshold)
        return None

    hit      = hits[0]
    doc_type = hit["payload"].get("document_type")
    if not doc_type:
        return None

    log.info("qdrant match: document_type=%r score=%.3f", doc_type, hit["score"])
    return {
        "document_type": doc_type,
        "template_id":   hit.get("template_id"),
        "score":         hit["score"],
    }


def _resolve_document_type(
    document_type: Optional[str],
    preclass: PreClassResult,
    image_path: Path,
    lines: list[TextLine],
    config: Config,
    vision_backend: OllamaVisionBackend,
) -> tuple[str, str, Optional[dict]]:
    if document_type:
        return document_type, "caller", None

    qdrant_hit = _try_vector_match(preclass, lines, config)
    if qdrant_hit:
        return qdrant_hit["document_type"], "qdrant", qdrant_hit

    inferred = _document_type_from_preclass(preclass)
    if inferred is not None:
        return inferred, "preclass_shortcut", None

    return identify(str(image_path), vision_backend), "vlm_identify", None


def _run(image_path: str | Path,
         config: Config,
         vision_backend: OllamaVisionBackend,
         document_type: str | None = None) -> dict:

    image_path = Path(image_path)
    log.info("processing %s", image_path.name)

    image  = load_image(image_path)
    engine = _get_engine(config)
    lines  = engine.extract(image)
    log.info("OCR extracted %d lines", len(lines))

    if not lines:
        log.warning("no text extracted — aborting")
        return {"error": "no text extracted", "source": None, "document_type": document_type}

    confidence_avg = _avg_confidence(lines)
    if confidence_avg < config.confidence_threshold:
        log.warning("confidence %.3f below threshold %s — aborting", confidence_avg, config.confidence_threshold)
        return {
            "error"         : "low confidence — document quality insufficient",
            "confidence_avg": round(confidence_avg, 4),
            "source"        : None,
            "document_type" : document_type,
        }

    preclass = preclassify(image, lines)
    log.info(
        "preclassifier: family=%s mrz_type=%s country=%s confidence=%.2f",
        preclass.doc_family, preclass.mrz_type, preclass.country_iso, preclass.confidence,
    )

    document_type, match_source, _ = _resolve_document_type(
        document_type, preclass, image_path, lines, config, vision_backend
    )
    log.info("document_type=%r resolved via source=%r", document_type, match_source)

    mrz = detect(lines)
    if mrz:
        log.info("MRZ detected: valid=%s", mrz.valid)
    else:
        log.debug("no MRZ detected")

    mrz_flags      : list[str]             = []
    mismatches     : list[dict]            = []
    template_fields: dict[str, str | None] = {}
    best_template  = None
    match_result   = None
    active_lines   = lines

    # ── Template matching ──────────────────────────────────────────────────
    issue_year = _extract_issue_year(lines)
    templates  = load_templates(document_type, issue_year) if document_type else []

    if templates:
        candidates = []
        for t in templates:
            aligned_img, was_aligned = align_to_template(image, t)
            if was_aligned:
                aligned_lines = engine.extract(aligned_img)
                mr = match_template(aligned_lines, t)
                candidates.append((t, mr, aligned_lines))
            else:
                mr = match_template(lines, t)
                candidates.append((t, mr, lines))
        best_template, match_result, active_lines = max(candidates, key=lambda x: x[1].match_score)
        log.info(
            "template match: score=%.3f matched=%s document=%s",
            match_result.match_score, match_result.matched, match_result.document_name,
        )

        for key, field_line_list in match_result.field_lines.items():
            ordered = _row_order(field_line_list)
            template_fields[key] = " ".join(l.text.strip() for l in ordered).strip() or None
        template_fields = normalize_fields(template_fields)

        # VLM fallback for null fields
        null_fields = [k for k, v in template_fields.items() if not v]
        if null_fields:
            log.debug("null fields after template match — asking VLM: %s", null_fields)
            output_for_vlm = _build_pipeline_output(document_type, active_lines, mrz, confidence_avg)
            try:
                template_schema = _schema_load(best_template)
            except Exception:
                template_schema = None
            spatial    = build_spatial_layout(active_lines)
            vlm_result = extract_with_vision(str(image_path), vision_backend, output_for_vlm, spatial, template_schema)
            for k in null_fields:
                if vlm_result.get("fields", {}).get(k):
                    template_fields[k] = vlm_result["fields"][k]
            template_fields = normalize_fields(template_fields)

    # ── MRZ comparison ─────────────────────────────────────────────────────
    if mrz:
        if not mrz.valid:
            mrz_flags.append("mrz_checksum_failed")
        if template_fields:
            mismatches = _compare_mrz(template_fields, mrz)
            if mismatches:
                mrz_flags.append("mrz_mismatch")

    log.info("final extracted fields: %d → %s", len(template_fields), list(template_fields.keys()))

    # ── Visualization (non-critical) ───────────────────────────────────────
    if match_result is not None:
        try:
            from service.ocr.visualizer import visualize_template_match
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vis_path  = Path(config.ocr_output_dir) / f"{image_path.stem}_template.png"
            visualize_template_match(image_bgr, active_lines, best_template, match_result, str(vis_path))
        except Exception as e:
            log.debug("template visualization skipped: %s", e)

    mrz_dict_out = _mrz_to_dict(mrz) if mrz else None
    verdict      = "unverifiable" if "mrz_mismatch" in mrz_flags else "verifiable"

    result = {
        "verdict"           : verdict,
        "match_score"       : match_result.match_score if match_result else None,
        "document_type"     : document_type,
        "document_name"     : match_result.document_name if match_result else None,
        "template_available": match_result is not None,
        "fields"            : template_fields,
        "mrz"               : mrz_dict_out,
        "unmatched_fields"  : match_result.unmatched_fields if match_result else [],
    }
    if mrz_flags:
        result["flags"] = mrz_flags
    if mismatches:
        result["mismatches"] = mismatches
    log.info("done processing %s", image_path.name)
    return result


def process(file_path: str | list,
            document_type: str | None = None) -> dict | list[dict]:
    config         = Config()
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, vision_backend, document_type) for fp in file_path]

    return _run(file_path, config, vision_backend, document_type)
