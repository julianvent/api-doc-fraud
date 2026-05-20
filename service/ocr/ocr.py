from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from service.ocr.backends import OllamaVisionBackend
from service.ocr.identifier import identify
from service.ocr.extractor import extract_with_vision

from .layout import build_spatial_layout
from .models import Config, PipelineOutput
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .mrz import detect
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


def _run(image_path: str | Path,
         config: Config,
         vision_backend: OllamaVisionBackend,
         document_type: str | None = None) -> dict:

    image_path = Path(image_path)
    print(f"\n[OCR] === processing {image_path.name} ===")

    image  = load_image(image_path)
    engine = _get_engine(config)
    print(f"[OCR] image loaded, engine={config.ocr_engine}")

    if document_type:
        print(f"[OCR] document_type='{document_type}' provided by caller — skipping VLM identify")
        lines = engine.extract(image)
    else:
        print(f"[OCR] no document_type provided — running VLM identify and OCR in parallel")
        with ThreadPoolExecutor(max_workers=2) as ex:
            id_future  = ex.submit(identify, str(image_path), vision_backend)
            ocr_future = ex.submit(engine.extract, image)
            document_type = id_future.result()
            lines         = ocr_future.result()
        print(f"[OCR] VLM identified document_type='{document_type}'")

    print(f"[OCR] OCR extracted {len(lines)} lines")

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

    template      = load_template(config.templates_dir, document_type)
    template_path = Path(config.templates_dir) / f"{document_type}.json"

    if template is not None:
        n_fields = len(iter_template_fields(template))
        print(f"[OCR] template FOUND at {template_path} ({n_fields} fields) → VLM guided by template")
    else:
        print(f"[OCR] no template at {template_path} → VLM free extraction")

    spatial_layout = build_spatial_layout(lines)
    result = extract_with_vision(str(image_path), vision_backend, output, spatial_layout, template)

    agent_fields = result.get("result", {}).get("fields", {})
    print(f"[OCR] final extracted fields: {len(agent_fields)} → {list(agent_fields.keys())}")

    print(f"[OCR] === done ===\n")
    return result


def process(file_path: str | list,
            document_type: str | None = None) -> dict | list[dict]:
    config         = Config()
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, vision_backend, document_type) for fp in file_path]

    return _run(file_path, config, vision_backend, document_type)
