from pathlib import Path

import cv2

from service.ocr.agent import analyze
from service.ocr.agent.base import LLMBackend
from service.ocr.agent.ollama import OllamaBackend
from service.ocr.visualizer import visualize_matplotlib

from .models import Config, PipelineOutput
from .engine import OCREngine, PaddleOCRAdapter, DotsOCRAdapter, DolphinOCRAdapter, load_image
from .language import filter_latin
from .mrz import detect


_ENGINES: dict[str, type[OCREngine]] = {
    "paddle":  PaddleOCRAdapter,
    "dots":    DotsOCRAdapter,
    "dolphin": DolphinOCRAdapter,
}

# Singleton por nombre de motor (permite comparar varios en el mismo proceso)
# sin reiniciar el servidor (util para el script de benchmarking)
_engine_cache: dict[str, OCREngine] = {}
_engine: OCREngine | None = None
_backend: OllamaBackend | None = None


# Factories

def _get_engine(config: Config) -> OCREngine:
    global _engine

    name = config.ocr_engine.lower().strip()

    if name not in _engine_cache:
        cls = _ENGINES.get(name)
        if cls is None:
            raise ValueError(
                f"Motor OCR desconocido: {name!r}. "
                f"Disponibles: {sorted(_ENGINES)}"
            )
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
    print(f"[OCR] Warmup — motor: {config.ocr_engine}")
    _get_engine(config)
    _get_backend(config)
    print(f"[OCR] Warmup completado")


# Internal helpers
def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def _run(image_path: str | Path, config: Config, backend: LLMBackend) -> dict:
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
            "source"        : None,
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
        raw_lines      = lines,
    )

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_dir = Path(config.ocr_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_result.png"
    visualize_matplotlib(image_bgr, output, str(output_path))

    return analyze(output, config, backend)


# Public API
def process(file_path: str | list) -> dict | list[dict]:
    config  = Config()
    backend = _get_backend(config)

    if isinstance(file_path, list):
        return [_run(fp, config, backend) for fp in file_path]

    return _run(file_path, config, backend)
