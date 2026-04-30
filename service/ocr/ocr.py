from pathlib import Path

import cv2

from service.ocr.agent import analyze
from service.ocr.agent.base import LLMBackend
from service.ocr.agent.ollama import OllamaBackend
from service.ocr.visualizer import visualize_matplotlib

from .models import Config, PipelineOutput
from .engine import PaddleOCRAdapter, load_image
from .language import filter_latin
from .mrz import detect



_engine  = None
_backend = None


def _get_engine(config: Config) -> PaddleOCRAdapter:
    global _engine
    if _engine is None:
        _engine = PaddleOCRAdapter(config)
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


def _run(
    image_path: str | Path,
    config: Config,
    backend: LLMBackend,
    output_dir: Path,
) -> dict:
    engine = _get_engine(config)
    image  = load_image(Path(image_path))
    lines  = engine.extract(image)

    if not lines:
        return {"error": "no text extracted", "source": None}

    confidence_avg = _avg_confidence(lines)

    if confidence_avg < config.confidence_threshold:
        return {
            "error"         : "low confidence — document quality insufficient",
            "confidence_avg": round(confidence_avg, 4),
            "source"        : None
        }

    english_lines = filter_latin(lines)
    english_text  = "\n".join(l.text for l in english_lines)
    mrz           = detect(english_lines)

    mrz_verified   = mrz if (mrz and mrz.valid)     else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None
    source         = (
        "mrz+gemma"         if mrz_verified   else
        "mrz_partial+gemma" if mrz_unverified else
        "gemma"
    )


    output = PipelineOutput(
        mrz_verified   = mrz_verified,
        mrz_unverified = mrz_unverified,
        english_lines  = english_lines,
        english_text   = english_text,
        source         = source,
        confidence_avg = round(confidence_avg, 4),
        raw_lines      = lines
    )
    image_bgr = cv2.imread(str(image_path))
    output_path = output_dir / f"{Path(image_path).stem}_result.png"
    visualize_matplotlib(image_bgr, output, str(output_path))

    return analyze(output, config, backend)


def process(
    file_path: str | Path | list,
    output_dir: Path | str,
) -> dict | list[dict]:
    config  = Config()
    backend = _get_backend(config)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(file_path, list):
        return [_run(fp, config, backend, out_path) for fp in file_path]

    return _run(file_path, config, backend, out_path)
