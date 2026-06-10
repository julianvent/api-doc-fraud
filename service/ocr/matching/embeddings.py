from typing import Optional

import requests

from service.logging_config import get_logger


log = get_logger(__name__)


def embed(
    text: str,
    ollama_url: str,
    model: str = "bge-m3",
    timeout: int = 30,
) -> Optional[list[float]]:
    """
    Calls Ollama embeddings API. Returns the embedding vector or None on failure.

    ollama_url should point at the embeddings endpoint, typically:
        http://localhost:11434/api/embeddings
    """
    if not text or not text.strip():
        return None

    try:
        response = requests.post(
            ollama_url,
            json    = {"model": model, "prompt": text},
            timeout = timeout,
        )
        response.raise_for_status()
        data   = response.json()
        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            log.warning("Ollama returned no embedding (model=%s)", model)
            return None
        return [float(x) for x in vector]
    except Exception as e:
        log.error("Ollama embeddings call failed: %s: %s", type(e).__name__, e)
        return None
