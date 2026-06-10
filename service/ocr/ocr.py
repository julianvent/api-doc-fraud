from pathlib import Path
from typing import Optional

from service.ocr.backends import OllamaVisionBackend
from service.ocr.identifier import identify
from service.ocr.extractor import extract_with_vision

from . import matching
from .layout import build_spatial_layout
from .models import Config, PipelineOutput, TextLine
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .mrz import detect
from .preclassifier import classify as preclassify, PreClassResult
from .templates import iter_template_fields, load_template


_ENGINES: dict[str, type[OCREngine]] = {
    "paddle":  PaddleOCRAdapter,
    "dots":    DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache   : dict[str, OCREngine]          = {}
_engine         : OCREngine | None              = None
_vision_backend : OllamaVisionBackend | None    = None


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


def _build_pipeline_output(document_type, lines, mrz, confidence_avg) -> PipelineOutput:
    mrz_verified   = mrz if (mrz and mrz.valid)     else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None
    source         = (
        "mrz+gemma"          if mrz_verified   else
        "mrz_partial+gemma"  if mrz_unverified else
        "gemma"
    )
    return PipelineOutput(
        document_type  = document_type,
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
    """
    Returns {"document_type", "template_id", "score"} on hit, None on miss / disabled / error.
    """
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
        print(f"[OCR] qdrant: no match above threshold={config.match_threshold}")
        return None

    hit      = hits[0]
    doc_type = hit["payload"].get("document_type")
    if not doc_type:
        return None

    print(
        f"[OCR] qdrant match: document_type='{doc_type}' "
        f"score={hit['score']:.3f} template_id={hit.get('template_id')}"
    )
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
    """
    Returns (document_type, source, qdrant_hit_or_None).
    source in {"caller", "qdrant", "preclass_shortcut", "vlm_identify"}.
    """
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
    print(f"\n[OCR] === processing {image_path.name} ===")

    image  = load_image(image_path)
    engine = _get_engine(config)
    print(f"[OCR] image loaded, engine={config.ocr_engine}")

    lines = engine.extract(image)
    print(f"[OCR] OCR extracted {len(lines)} lines")

    preclass = preclassify(image, lines)
    print(
        f"[OCR] preclassifier: family={preclass.doc_family} mrz_type={preclass.mrz_type} "
        f"country={preclass.country_iso} confidence={preclass.confidence:.2f}"
    )

    document_type, match_source, qdrant_hit = _resolve_document_type(
        document_type, preclass, image_path, lines, config, vision_backend
    )
    print(f"[OCR] document_type='{document_type}' resolved via source='{match_source}'")

    if not lines:
        print(f"[OCR] no text extracted — aborting")
        return {"error": "no text extracted", "source": None, "document_type": document_type}

    confidence_avg = _avg_confidence(lines)
    print(f"[OCR] avg OCR confidence: {confidence_avg:.3f}")

    if confidence_avg < config.confidence_threshold:
        print(f"[OCR] confidence below threshold ({config.confidence_threshold}) — aborting")
        return {
            "error"         : "low confidence — document quality insufficient",
            "confidence_avg": round(confidence_avg, 4),
            "source"        : None,
            "document_type" : document_type,
        }

    mrz = detect(lines)
    if mrz:
        print(f"[OCR] MRZ detected: valid={mrz.valid}")
    else:
        print(f"[OCR] no MRZ detected")

    output = _build_pipeline_output(document_type, lines, mrz, confidence_avg)

    country_iso = preclass.country_iso
    template    = load_template(
        document_type,
        country_iso=country_iso,
        templates_dir=config.templates_dir,
    )

    if template is not None:
        n_fields = len(iter_template_fields(template))
        print(
            f"[OCR] template FOUND on disk (document_type={document_type}, "
            f"country_iso={template.country_iso}, edition={template.edition}, "
            f"{n_fields} fields) → VLM guided by template"
        )
    else:
        print(
            f"[OCR] no template on disk for document_type={document_type!r} "
            f"(country_iso={country_iso!r}) → VLM free extraction"
        )

    spatial_layout = build_spatial_layout(lines)
    result = extract_with_vision(str(image_path), vision_backend, output, spatial_layout, template)

    agent_fields = result.get("result", {}).get("fields", {})
    print(f"[OCR] final extracted fields: {len(agent_fields)} → {list(agent_fields.keys())}")

    if isinstance(result, dict):
        result["preclass"] = {
            "doc_family":   preclass.doc_family,
            "country_iso":  preclass.country_iso,
            "mrz_type":     preclass.mrz_type,
            "has_face":     preclass.has_face,
            "aspect_class": preclass.aspect_class,
            "confidence":   preclass.confidence,
        }
        result["match_source"] = match_source
        if qdrant_hit:
            result["matched_template_id"] = qdrant_hit["template_id"]
            result["match_score_vector"] = qdrant_hit["score"]

    print(f"[OCR] === done ===\n")
    return result


def process(file_path: str | list,
            document_type: str | None = None) -> dict | list[dict]:
    config         = Config()
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, vision_backend, document_type) for fp in file_path]

    return _run(file_path, config, vision_backend, document_type)
