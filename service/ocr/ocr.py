from pathlib import Path

import cv2
from rapidfuzz import fuzz

from service.ocr.agent import analyze, fill_missing_fields
from service.ocr.agent.base import LLMBackend
from service.ocr.agent.ollama import OllamaBackend
from service.ocr.visualizer import visualize_matplotlib, visualize_template_match, visualize_agent_extraction
from service.ocr.template_matcher import load_templates, match as match_template, align_to_template
from service.ocr.preclassifier import classify as preclassify
from service.ocr import matching

from .models import Config, PipelineOutput
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .language import filter_latin
from .normalizer import normalize_fields, normalize_date
from .mrz import detect


_ENGINES: dict[str, type[OCREngine]] = {
    "paddle":  PaddleOCRAdapter,
    "dots":    DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache : dict[str, OCREngine] = {}
_engine       : OCREngine | None     = None
_backend      : OllamaBackend | None = None


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


def _get_backend(config: Config) -> OllamaBackend:
    global _backend
    if _backend is None:
        _backend = OllamaBackend(config.ollama_url, config.ollama_model)
    return _backend


def warmup() -> None:
    config = Config()
    _get_engine(config)
    _get_backend(config)


def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def _extract_issue_year(lines: list) -> int | None:
    """Escanea las líneas del probe OCR buscando fechas candidatas a date_of_issue.
    Heurística: de todas las fechas reconocidas que no sean futuras, toma la más
    reciente (la de nacimiento suele ser la más antigua, la de expedición la segunda
    más reciente, la de caducidad suele ser futura).
    """
    from datetime import datetime
    current_year = datetime.now().year
    found: list[int] = []
    for line in lines:
        normalized = normalize_date(line.text.strip())
        if not normalized or normalized == line.text.strip():
            continue  # normalize_date devolvió el original → no reconoció la fecha
        try:
            dt = datetime.strptime(normalized, "%d/%m/%Y")
            if dt.year <= current_year:
                found.append(dt.year)
        except ValueError:
            continue
    if not found:
        return None
    # Excluye el mínimo (fecha de nacimiento) y toma el mayor restante
    found_sorted = sorted(set(found))
    candidates = found_sorted[1:] if len(found_sorted) > 1 else found_sorted
    year = max(candidates)
    print(f"[PROBE] fechas encontradas={found_sorted} → issue_year estimado={year}")
    return year



def _mrz_to_dict(mrz) -> dict:
    return normalize_fields({
        "surname"        : mrz.surname,
        "given_names"    : mrz.given_names,
        "country"        : mrz.country,
        "date_of_birth"  : mrz.date_of_birth,
        "date_of_expiry" : mrz.expiry_date,
        "document_number": mrz.document_number,
        "sex"            : mrz.sex,
    })


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


def _norm(value: str) -> str:
    return (normalize_date(str(value)) or str(value)).strip().upper().replace(" ", "")


_MRZ_COMPARABLE_FIELDS = {"surname", "given_names", "date_of_birth", "expiry_date", "document_number", "sex", "country"}


def _compare_mrz(template_fields: dict, mrz) -> list[dict]:
    """Compara los campos del template contra el MRZ para las claves que existen en ambos."""
    mrz_dict   = _mrz_to_dict(mrz)
    mismatches = []
    common     = set(template_fields) & set(mrz_dict) & _MRZ_COMPARABLE_FIELDS
    print(f"[MRZ-CMP] campos comunes a comparar: {common}")
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
            mismatches.append({
                "field"         : key,
                "template_value": tv,
                "mrz_value"     : mv,
                "similarity"    : similarity,
            })
    return mismatches





def _save_visualization(image, output: PipelineOutput, image_path: Path, config: Config) -> None:
    image_bgr  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_dir = Path(config.ocr_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_matplotlib(image_bgr, output, str(output_dir / f"{image_path.stem}_result.png"))


def _build_pipeline_output(english_lines, english_text, mrz, lines,
                            confidence_avg,
                            template_available=False,
                            template_match_score=None) -> PipelineOutput:
    mrz_verified   = mrz if (mrz and mrz.valid)     else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None
    source         = (
        "template+mrz+gemma"    if (template_available and mrz_verified)   else
        "template+mrz_partial"  if (template_available and mrz_unverified) else
        "template+gemma"        if template_available                       else
        "mrz+gemma"             if mrz_verified                            else
        "mrz_partial+gemma"     if mrz_unverified                          else
        "gemma"
    )
    return PipelineOutput(
        mrz_verified         = mrz_verified,
        mrz_unverified       = mrz_unverified,
        english_lines        = english_lines,
        english_text         = english_text,
        source               = source,
        confidence_avg       = round(confidence_avg, 4),
        raw_lines            = lines,
        template_available   = template_available,
        template_match_score = template_match_score,
    )


_MRZ_TYPE_TO_DOC: dict[str, str] = {"TD3": "passport", "MRV-A": "visa", "MRV-B": "visa"}

# Cuando Qdrant no tiene un template específico, estos doc_family
# mapean a un template genérico (si existe en templates/).
_DOC_FAMILY_FALLBACK: dict[str, str] = {
    "proof_of_address": "proof_of_address",
    "identity_card":    "identity_card",
    "identity_photo":   "identity_photo",
}


def _resolve_document_type(image, lines: list, config: Config) -> str | None:
    """Identify document type via preclassifier → Qdrant → MRZ shortcut → doc_family fallback."""
    preclass = preclassify(image, lines)
    print(
        f"[OCR] preclassifier: family={preclass.doc_family} "
        f"mrz_type={preclass.mrz_type} country={preclass.country_iso} "
        f"confidence={preclass.confidence:.2f}"
    )

    # 1 ── Qdrant vector match (específico por emisor/país/edición)
    if not config.disable_vector_match and matching.is_available():
        query_text = matching.serialize_for_query(preclass, lines)
        vector     = matching.embed(query_text, config.embedding_url, config.embedding_model)
        if vector:
            filters = {
                "doc_family":  preclass.doc_family if preclass.doc_family != "unknown" else None,
                "mrz_type":    preclass.mrz_type,
                "country_iso": preclass.country_iso,
            }
            hits = matching.search(
                config.qdrant_url, config.qdrant_collection, vector,
                filters=filters, limit=1, score_threshold=config.match_threshold,
            )
            if hits:
                doc_type = hits[0]["payload"].get("document_type")
                if doc_type:
                    print(f"[OCR] qdrant match: {doc_type} score={hits[0]['score']:.3f}")
                    return doc_type

    # 2 ── Shortcut por tipo de MRZ (passport, visa)
    if preclass.mrz_type:
        doc_type = _MRZ_TYPE_TO_DOC.get(preclass.mrz_type)
        if doc_type:
            print(f"[OCR] mrz shortcut: {preclass.mrz_type} → {doc_type}")
            return doc_type

    # 3 ── Fallback genérico por familia (proof_of_address, identity_card, etc.)
    #      Usa el template genérico si existe en templates/, en lugar de ir directo al agente.
    if preclass.doc_family in _DOC_FAMILY_FALLBACK:
        doc_type = _DOC_FAMILY_FALLBACK[preclass.doc_family]
        print(f"[OCR] doc_family fallback: {preclass.doc_family} → {doc_type}")
        return doc_type

    print("[OCR] document type could not be resolved")
    return None


def _run(image_path: str | Path,
         config: Config,
         backend: LLMBackend,
         document_type: str | None = None) -> dict:

    engine     = _get_engine(config)
    image_orig = load_image(Path(image_path))

    # ── pasada 1: OCR rápido para identificar tipo de documento ───────────
    # Solo se ejecuta cuando el caller no provee document_type.
    if not document_type:
        lines_probe = engine.extract(image_orig)
        if not lines_probe:
            return {"error": "no text extracted", "source": None}
        confidence_avg = _avg_confidence(lines_probe)
        if confidence_avg < config.confidence_threshold:
            return {
                "error"         : "low confidence — document quality insufficient",
                "confidence_avg": round(confidence_avg, 4),
                "source"        : None,
            }
        document_type = _resolve_document_type(image_orig, lines_probe, config)
        issue_year    = _extract_issue_year(lines_probe)
    else:
        issue_year = None

    templates = load_templates(document_type, issue_year) if document_type else []

    # ── homography alignment ───────────────────────────────────────────────
    image = image_orig
    if templates:
        image, aligned_ok = align_to_template(image_orig, templates[0])
        print(f"[OCR] homography alignment: {'ok' if aligned_ok else 'skipped (no ref image or too few keypoints)'}")

    # ── pasada 2: OCR final sobre imagen alineada ─────────────────────────
    lines = engine.extract(image)

    if not lines:
        return {"error": "no text extracted", "source": None}

    confidence_avg = _avg_confidence(lines)

    if confidence_avg < config.confidence_threshold:
        return {
            "error"         : "low confidence — document quality insufficient",
            "confidence_avg": round(confidence_avg, 4),
            "source"        : None,
        }

    # ── template path ──────────────────────────────────────────────────────
    if document_type and templates:
        best_template, match_result = max(
            ((t, match_template(lines, t)) for t in templates),
            key=lambda pair: pair[1].match_score,
        )
        print(f"[TMPL] document_type={document_type!r} match_score={match_result.match_score:.3f} matched={match_result.matched}")

        if not match_result.matched:
            return {
                "verdict"            : "unverifiable",
                "flags"              : ["layout_mismatch"],
                "match_score"        : match_result.match_score,
                "document_type"      : document_type,
                "template_available" : True,
                "message"            : "document layout does not match expected template",
            }

        template_fields: dict[str, str | None] = {}
        for key, field_lines in match_result.field_lines.items():
            if not field_lines:
                template_fields[key] = None
                continue
            ordered = _row_order(field_lines)
            template_fields[key] = " ".join(l.text.strip() for l in ordered).strip()
        template_fields = normalize_fields(template_fields)
        print(f"[TMPL] extracted fields: {template_fields}")

        english_lines = filter_latin(lines)

        null_fields = [k for k, v in template_fields.items() if not v]
        if null_fields:
            print(f"[TMPL] null fields → asking VLM: {null_fields}")
            filled = fill_missing_fields(english_lines, backend, null_fields)
            for k, v in filled.items():
                if v:
                    template_fields[k] = v
            template_fields = normalize_fields(template_fields)
            print(f"[TMPL] fields after VLM fill: {template_fields}")

        mrz = detect(lines)
        print(f"[MRZ] detected={mrz is not None} valid={mrz.valid if mrz else 'N/A'}")
        if mrz:
            print(f"[MRZ] raw → surname={mrz.surname!r} given={mrz.given_names!r} "
                  f"dob={mrz.date_of_birth!r} expiry={mrz.expiry_date!r} "
                  f"num={mrz.document_number!r} country={mrz.country!r}")

        mrz_flags: list[str] = []
        mismatches: list[dict] = []
        if mrz:
            if not mrz.valid:
                mrz_flags.append("mrz_checksum_failed")
                print("[MRZ-CMP] MRZ checksum invalid — comparando de todas formas")
            mismatches = _compare_mrz(template_fields, mrz)
            print(f"[MRZ-CMP] mismatches found: {len(mismatches)}")
            if mismatches:
                mrz_flags.append("mrz_mismatch")
        else:
            print("[MRZ-CMP] skipped: no MRZ detected")

        output = _build_pipeline_output(
            lines,
            "\n".join(l.text for l in lines),
            mrz, lines, confidence_avg,
            template_available   = True,
            template_match_score = match_result.match_score,
        )
        _save_visualization(image, output, Path(image_path), config)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        template_vis_path = Path(config.ocr_output_dir) / f"{Path(image_path).stem}_template.png"
        visualize_template_match(image_bgr, lines, best_template, match_result, str(template_vis_path))

        mrz_dict_out = _mrz_to_dict(mrz) if mrz else None

        verdict = "unverifiable" if "mrz_mismatch" in mrz_flags else "verifiable"
        result  = {
            "verdict"           : verdict,
            "match_score"       : match_result.match_score,
            "document_type"     : document_type,
            "document_name"     : match_result.document_name,
            "template_available": True,
            "fields"            : template_fields,
            "mrz"               : mrz_dict_out,
            "unmatched_fields"  : match_result.unmatched_fields,
        }
        if mrz_flags:
            result["flags"] = mrz_flags
        if mismatches:
            result["mismatches"] = mismatches
        return result

    # ── fallback path ──────────────────────────────────────────────────────
    english_lines = filter_latin(lines)
    english_text  = "\n".join(l.text for l in english_lines)
    mrz           = detect(english_lines)

    output = _build_pipeline_output(
        english_lines, english_text, mrz, lines, confidence_avg
    )

    _save_visualization(image, output, Path(image_path), config)

    result                       = analyze(output, config, backend)
    result["template_available"] = False

    agent_fields = result.get("result", {}).get("fields", {})
    if agent_fields:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        agent_vis_path = Path(config.ocr_output_dir) / f"{Path(image_path).stem}_agent.png"
        visualize_agent_extraction(image_bgr, lines, agent_fields, str(agent_vis_path))

    return result


def process(file_path: str | list,
            document_type: str | None = None) -> dict | list[dict]:
    config  = Config()
    backend = _get_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, backend, document_type) for fp in file_path]

    return _run(file_path, config, backend, document_type)