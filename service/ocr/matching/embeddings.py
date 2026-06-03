from typing import Optional

import requests


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
            print(f"[embeddings] Ollama returned no embedding (model={model})")
            return None
        return [float(x) for x in vector]
    except Exception as e:
        print(f"[embeddings] Ollama call failed: {type(e).__name__}: {e}")
        return None
