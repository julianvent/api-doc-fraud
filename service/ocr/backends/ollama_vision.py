import base64
import os
import requests

from service.logging_config import get_logger
from .base import VisionBackend


log = get_logger(__name__)

_DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "600"))


class VLMTimeout(Exception):
    """Raised when the VLM backend does not respond within the configured timeout."""


class VLMUnavailable(Exception):
    """Raised when the VLM backend is unreachable or returns a non-2xx status."""


class OllamaVisionBackend(VisionBackend):

    def __init__(self, url: str, model: str, timeout: int = _DEFAULT_TIMEOUT):
        self._url     = os.getenv("OLLAMA_VISION_URL", url)
        self._model   = os.getenv("OLLAMA_VISION_MODEL", model)
        self._timeout = timeout

    def describe(self, image_path: str, prompt: str, max_tokens: int = 50) -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")

        try:
            response = requests.post(
                self._url,
                json={
                    "model"  : self._model,
                    "prompt" : prompt,
                    "images" : [image_b64],
                    "stream" : False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                        "num_ctx"    : 16384,
                    },
                },
                timeout=self._timeout,
            )
        except requests.Timeout as e:
            raise VLMTimeout(
                f"Ollama VLM did not respond within {self._timeout}s"
            ) from e
        except requests.ConnectionError as e:
            raise VLMUnavailable(f"Ollama VLM unreachable at {self._url}") from e

        if not response.ok:
            raise VLMUnavailable(
                f"Ollama VLM returned HTTP {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        text = body.get("response", "").strip()
        if not text:
            log.warning("VLM empty response. Ollama body: %s", body)
        return text
