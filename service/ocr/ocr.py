from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from service.ocr.agent.agent import _load_template, _build_spatial_layout, _iter_template_fields
from service.ocr.agent.base import LLMBackend
from service.ocr.agent.ollama import OllamaBackend
from service.ocr.backends import OllamaVisionBackend
from service.ocr.visualizer import visualize_matplotlib, visualize_agent_extraction
from service.ocr.identifier import identify
from service.ocr.extractor import extract_with_vision

from .models import Config, PipelineOutput
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .mrz import detect


_ENGINES: dict[str, type[OCREngine]] = {
    "paddle":  PaddleOCRAdapter,
    "dots":    DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

_engine_cache   : dict[str, OCREngine]          = {}
_engine         : OCREngine | None              = None
_backend        : OllamaBackend | None          = None
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


def _get_backend(config: Config) -> OllamaBackend:
    global _backend
    if _backend is None:
        _backend = OllamaBackend(config.ollama_url, config.ollama_model)
    return _backend


def _get_vision_backend(config: Config) -> OllamaVisionBackend:
    global _vision_backend
    if _vision_backend is None:
        _vision_backend = OllamaVisionBackend(config.ollama_vision_url, config.ollama_vision_model)
    return _vision_backend


def warmup() -> None:
    config = Config()
    _get_engine(config)
    _get_backend(config)
    _get_vision_backend(config)


def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def _save_visualization(image, output: PipelineOutput, image_path: Path, config: Config) -> None:
    image_bgr  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_dir = Path(config.ocr_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_matplotlib(image_bgr, output, str(output_dir / f"{image_path.stem}_result.png"))


def _build_pipeline_output(document_type, lines, english_text, mrz, raw_lines,
                           confidence_avg) -> PipelineOutput:
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
        english_text   = english_text,
        source         = source,
        confidence_avg = round(confidence_avg, 4),
        raw_lines      = raw_lines,
    )


def _run(image_path: str | Path,
         config: Config,
         backend: LLMBackend,
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

    english_text = "\n".join(l.text for l in lines)
    mrz          = detect(lines)
    if mrz:
        print(f"[OCR] MRZ detected: valid={mrz.valid}")
    else:
        print(f"[OCR] no MRZ detected")

    output = _build_pipeline_output(
        document_type, lines, english_text, mrz, lines, confidence_avg
    )

    _save_visualization(image, output, image_path, config)
    print(f"[OCR] base visualization saved → {image_path.stem}_result.png")

    template      = _load_template(config.templates_dir, document_type)
    template_path = Path(config.templates_dir) / f"{document_type}.json"

    if template is not None:
        n_fields = len(_iter_template_fields(template))
        print(f"[OCR] template FOUND at {template_path} ({n_fields} fields) → VLM guided by template")
    else:
        print(f"[OCR] no template at {template_path} → VLM free extraction")

    spatial_layout = _build_spatial_layout(lines)
    result = extract_with_vision(str(image_path), vision_backend, output, spatial_layout, template)

    agent_fields = result.get("result", {}).get("fields", {})
    print(f"[OCR] final extracted fields: {len(agent_fields)} → {list(agent_fields.keys())}")

    if agent_fields:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        agent_vis_path = Path(config.ocr_output_dir) / f"{image_path.stem}_agent.png"
        visualize_agent_extraction(image_bgr, lines, agent_fields, str(agent_vis_path))
        print(f"[OCR] agent visualization saved → {agent_vis_path.name}")

    print(f"[OCR] === done ===\n")
    return result


def process(file_path: str | list,
            document_type: str | None = None) -> dict | list[dict]:
    config         = Config()
    backend        = _get_backend(config)
    vision_backend = _get_vision_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, backend, vision_backend, document_type) for fp in file_path]

    return _run(file_path, config, backend, vision_backend, document_type)
