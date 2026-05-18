import base64
import json
import re
import requests
from pathlib import Path

from PIL import Image as PILImage

from .model import TemplateConfig
from .prompt import _build_prompt


def _load_image(path: str | Path, max_side: int = 1_600) -> PILImage.Image:
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return img


def _encode_image(path: str | Path) -> str:
    """Carga la imagen, la redimensiona si es necesario y la devuelve en base64."""
    img = _load_image(path)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _slugify(text: str) -> str:
    latin = re.sub(r"[^\x00-\x7F]+", " ", text).strip()
    text  = latin if latin else text
    text  = re.sub(r"[^a-z0-9\s]", "", text.lower().strip())
    return re.sub(r"\s+", "_", text)[:40] or "field"


def _unique_key(base: str, existing: set) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _call_llm(
    img_path: str | Path,
    ollama_url: str,
    ollama_model: str,
    document_type: str,
) -> list[dict]:
    prompt      = _build_prompt(document_type)
    encoded_img = _encode_image(img_path)

    try:
        response = requests.post(
            ollama_url,
            json={
                "model":       ollama_model,
                "prompt":      prompt,
                "images":      [encoded_img],
                "stream":      False,
                "temperature": 0,
            },
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except Exception as e:
        print(f"  [TemplateOCR/LLM] Error: {e}")
        return []

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                print("  [TemplateOCR/LLM] No se pudo parsear la respuesta")
                return []
        else:
            return []

    return data.get("fields", []) if isinstance(data, dict) else []


def extract_template(
    img_path: str | Path,
    config: "TemplateConfig | None" = None,
    document_type: str = "generic",
) -> dict:
    """
    Identifica los campos de un documento a partir de su imagen
    Devuelve una lista de {key, label, type}
    """
    if config is None:
        config = TemplateConfig()

    print(f"  [TemplateOCR] Enviando imagen a {config.ollama_model}...")
    raw_fields = _call_llm(img_path, config.ollama_url, config.ollama_model, document_type)

    if not raw_fields:
        print("  [TemplateOCR] Warning: el modelo no devolvió campos")
        return {"fields": [], "n_fields": 0}

    # Normalizar y generar keys en Python
    used_keys : set        = set()
    fields    : list[dict] = []

    for f in raw_fields:
        label = str(f.get("label", "")).strip()
        ftype = str(f.get("type",  "text")).strip()
        if not label:
            continue

        # Usar el key que dio el LLM como base, pero regenerarlo limpio en Python
        llm_key  = str(f.get("key", "")).strip()
        base_key = _slugify(llm_key if llm_key else label)
        key      = _unique_key(base_key, used_keys)
        used_keys.add(key)

        fields.append({"key": key, "label": label, "type": ftype})

    print(f"  [TemplateOCR] {len(fields)} campos identificados")
    return {"fields": fields, "n_fields": len(fields)}