import base64
import os
import requests

from .base import VisionBackend


class OllamaVisionBackend(VisionBackend):

    def __init__(self, url: str, model: str):
        self._url   = os.getenv("OLLAMA_VISION_URL", url)
        self._model = os.getenv("OLLAMA_VISION_MODEL", model)

    def describe(self, image_path: str, prompt: str, max_tokens: int = 50) -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")

        response = requests.post(self._url, json={
            "model"  : self._model,
            "prompt" : prompt,
            "images" : [image_b64],
            "stream" : False,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
                "num_ctx"    : 10000
            }
        })
        if not response.ok:
            print(
                f"[VLM] Ollama HTTP {response.status_code} on {self._url} "
                f"(model={self._model}, prompt_chars={len(prompt)}, "
                f"image_b64_chars={len(image_b64)}, num_predict={max_tokens})"
            )
            print(f"[VLM] Ollama body: {response.text[:2000]}")
        response.raise_for_status()
        body = response.json()
        text = body.get("response", "").strip()
        if not text:
            print(f"[VLM] empty response. Ollama returned: {body}")
        return text
