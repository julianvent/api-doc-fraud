from pathlib import Path
import cv2
from rapidfuzz import fuzz

from api.v1.schema.verify import Identity

from service.ocr.backends.ollama_vision import OllamaVisionBackend
from service.ocr.template_matcher import (
    load_templates,
    match as match_template,
    align_to_template,
)
from service.ocr.extractor import extract_with_vision
from service.template_ocr.schema import load_template as _schema_load
from service.ocr.visualizer import visualize_matplotlib, visualize_template_match

from .models import Config, PipelineOutput
from .engine import (
    OCREngine,
    PaddleOCRAdapter,
    DotsOCRAdapter,
    DolphinOCRAdapter,
    load_image,
)
from .normalizer import normalize_fields, normalize_date
from .mrz import detect
from .preclassifier import classify as preclassify

_ENGINES: dict[str, type[OCREngine]] = {
    "paddle": PaddleOCRAdapter,
    "dots": DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache: dict[str, OCREngine] = {}
_vision_backend: OllamaVisionBackend | None = None


def _get_engine(config: Config) -> OCREngine:
    name = config.ocr_engine.lower().strip()
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
        _vision_backend = OllamaVisionBackend(
            config.ollama_vision_url, config.ollama_vision_model
        )
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
    """Scan probe OCR lines for a date_of_issue candidate.
    Heuristic: collect all non-future dates, drop the oldest (birth date),
    take the largest remaining year (issue date). Expiry dates are future and ignored.
    """
    from datetime import datetime

    current_year = datetime.now().year
    found: list[int] = []
    for line in lines:
        normalized = normalize_date(line.text.strip())
        if not normalized or normalized == line.text.strip():
            continue  # normalize_date returned original — not a recognized date
        try:
            dt = datetime.strptime(normalized, "%d/%m/%Y")
            if dt.year <= current_year:
                found.append(dt.year)
        except ValueError:
            continue
    if not found:
        return None
    found_sorted = sorted(set(found))
    # drop the oldest date (birth) and take the largest remaining
    candidates = found_sorted[1:] if len(found_sorted) > 1 else found_sorted
    year = max(candidates)
    print(f"[PROBE] dates found={found_sorted} → estimated issue_year={year}")
    return year


def _spatial_layout(lines: list) -> str:
    """Format OCR lines as a spatial text layout for the VLM prompt."""
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


_ROW_GROUP_THRESHOLD = 0.03
MRZ_MATCH_THRESHOLD = 100


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


def _mrz_to_dict(mrz) -> dict:
    return normalize_fields(
        {
            "surname": mrz.surname,
            "given_names": mrz.given_names,
            "country": mrz.country,
            "date_of_birth": mrz.date_of_birth,
            "expiry_date": mrz.expiry_date,
            "document_number": mrz.document_number,
            "sex": mrz.sex,
        }
    )


def _compare_mrz(template_fields: dict, mrz) -> list[dict]:
    mrz_dict = _mrz_to_dict(mrz)
    mismatches = []
    common = set(template_fields) & set(mrz_dict) & _MRZ_COMPARABLE_FIELDS
    print(f"[MRZ-CMP] common fields to compare: {common}")
    for key in common:
        tv = template_fields.get(key)
        mv = mrz_dict.get(key)
        if not tv or not mv:
            print(f"[MRZ-CMP] skip '{key}': tv={tv!r} mv={mv!r}")
            continue
        tv_n, mv_n = _norm(tv), _norm(mv)
        similarity = fuzz.ratio(tv_n, mv_n)
        status = "MISMATCH" if similarity < MRZ_MATCH_THRESHOLD else "ok"
        print(f"[MRZ-CMP] {status} '{key}': ocr={tv_n!r} mrz={mv_n!r} sim={similarity}")
        if similarity < MRZ_MATCH_THRESHOLD:
            mismatches.append(
                {
                    "field": key,
                    "document_value": tv,
                    "mrz_value": mv,
                }
            )
    return mismatches


_MRZ_TYPE_TO_DOC: dict[str, str] = {"TD3": "passport", "MRV-A": "visa", "MRV-B": "visa"}

# When Qdrant has no specific issuer template, map doc_family to a generic one (if it exists in templates/).
_DOC_FAMILY_FALLBACK: dict[str, str] = {
    "proof_of_address": "proof_of_address",
    "identity_card": "identity_card",
    "identity_photo": "identity_photo",
}


def _resolve_document_type(image, lines: list) -> str | None:
    """Identify document type: preclassifier → MRZ shortcut → doc_family fallback.
    Vector search is disabled by default; enable with DISABLE_VECTOR_MATCH=0.
    """
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


def _run(
    image_path: str | Path,
    config: Config,
    vision_backend: OllamaVisionBackend,
    identity_packet: Identity,
    document_type: str | None = None,
) -> dict:

    engine = _get_engine(config)
    image_orig = load_image(Path(image_path))

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

    image = image_orig
    if templates:
        image, aligned_ok = align_to_template(image_orig, templates[0])
        print(
            f"[OCR] homography alignment: {'ok' if aligned_ok else 'skipped (no ref image or too few keypoints)'}"
        )

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

    # ── template path (primary: OCR bbox matching) ─────────────────────────
    if document_type and templates:
        best_template, template_match_confidence = max(
            ((t, match_template(lines, t)) for t in templates),
            key=lambda pair: pair[1].match_score,
        )
        print(
            f"[TMPL] document_type={document_type!r} match_score={template_match_confidence.match_score:.3f} matched={template_match_confidence.matched}"
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

        # VLM fallback: fill fields that OCR could not read
        null_fields = [k for k, v in template_fields.items() if not v]
        if null_fields:
            print(f"[TMPL] null fields → asking VLM: {null_fields}")
            output_for_vlm = _build_pipeline_output(
                document_type, mrz, lines, ocr_confidence
            )
            try:
                template_schema = _schema_load(best_template)
            except Exception:
                template_schema = None
            spatial = _spatial_layout(lines)
            vlm_result = extract_with_vision(
                str(image_path),
                vision_backend,
                output_for_vlm,
                spatial,
                template_schema,
            )
            vlm_fields = vlm_result.get("fields", {})
            for k in null_fields:
                if vlm_fields.get(k):
                    template_fields[k] = vlm_fields[k]
            template_fields = normalize_fields(template_fields)
            print(f"[TMPL] fields after VLM fill: {template_fields}")

        mrz_flags: list[str] = []
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

        # Validation with identity packet
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
                image_bgr, lines, best_template, template_match_confidence, str(vis_path)
            )
        except Exception as e:
            print(
                f"[VIS] template visualization failed (non-critical): {type(e).__name__}: {e}"
            )

        mrz_dict_out = _mrz_to_dict(mrz) if mrz else None

        result = {
            "template_match_confidence": template_match_confidence.match_score,
            "ocr_confidence": ocr_confidence,
            "document_type": document_type,
            "document_name": template_match_confidence.document_name,
            "template_available": True,
            "fields": template_fields,
            "mrz": mrz_dict_out,
            "unmatched_fields": template_match_confidence.unmatched_fields,
            "flags": mrz_flags,
            "mrz_mismatches": mrz_mismatches,
            "identity_mismatches": identity_mismatches,
        }

        return result

    # ── fallback path (no template: full VLM extraction) ───────────────────
    output = _build_pipeline_output(document_type, mrz, lines, ocr_confidence)
    spatial = _spatial_layout(lines)
    _save_visualization(image, output, Path(image_path), config)
    return extract_with_vision(str(image_path), vision_backend, output, spatial, None)


def _compare_identity(template_fields: dict, identity: Identity) -> list[dict]:
    identity_dict = identity.model_dump()
    mismatches = []
    common = set(template_fields) & set(identity_dict) & _IDENTITY_COMPARABLE_FIELDS
    for key in common:
        template_value = template_fields.get(key)
        identity_value = identity_dict.get(key)

        if not template_value or not identity_value:
            continue

        template_value_norm = _norm(template_value)
        identity_value_norm = _norm(identity_value)

        if not template_value_norm == identity_value_norm:
            mismatches.append(
                {
                    "field": key,
                    "document_value": template_value,
                    "packet_value": identity_value,
                }
            )

    return mismatches


def process(
    file_path: str | list,
    identity: Identity,
    document_type: str | None = None,
) -> dict | list[dict]:
    config = Config()
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [
            _run(fp, config, vision_backend, identity, document_type)
            for fp in file_path
        ]

    return _run(file_path, config, vision_backend, identity, document_type)
