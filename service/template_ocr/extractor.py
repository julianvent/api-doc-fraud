import base64
import json
import os
import re
import requests
from pathlib import Path

from PIL import Image as PILImage

from .model import TemplateConfig
from .prompt import _build_prompt


_DEFAULT_MAX_SIDE = int(os.getenv("TEMPLATE_IMAGE_MAX_SIDE", "1024"))


def _load_image(path: str | Path, max_side: int = _DEFAULT_MAX_SIDE) -> PILImage.Image:
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
    timeout: int = 600,
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
                "temperature": 0.0,
            },
            timeout=timeout,
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

    if not isinstance(data, dict):
        return {}
    # Devolver las categorias tal como vienen del modelo
    return {
        "personal":    data.get("personal", []),
        "document":    data.get("document", []),
        "fingerprint": data.get("fingerprint") or {},
        "validators":  data.get("validators") or {},
    }


# Keys que deben coincidir exactamente con los campos para validar el MRZ
_MRZ_KEYS = {
    "surname", "given_names", "givennames", "givenname",
    "birth_date", "birthdate", "dateofbirth", "date_of_birth",
    "expiry_date", "expirydate", "dateofexpiry",
    "document_number", "documentnumber", "docnumber",
    "sex", "country", "nationality",
}

# Mapeo al key estandar correcto
_MRZ_KEY_MAP = {
    "givennames":     "given_names",
    "givenname":      "given_names",
    "birthdate":      "birth_date",
    "dateofbirth":    "birth_date",
    "date_of_birth":    "birth_date",
    "expirydate":     "expiry_date",
    "dateofexpiry":   "expiry_date",
    "documentnumber": "document_number",
    "docnumber":      "document_number",
}

def _normalize_category(raw: list, category: str, used_keys: set) -> list[dict]:
    result = []
    for f in raw:
        label = str(f.get("label", "")).strip()
        ftype = str(f.get("type", "text")).strip()
        if not label:
            continue

        llm_key = str(f.get("key", "")).strip().lower()

        if llm_key in _MRZ_KEYS:
            # Campo MRZ — aplicar normalización estricta al key
            base_key = _MRZ_KEY_MAP.get(llm_key, llm_key)
        else:
            # Campo normal — derivar key del label
            base_key = _slugify(label)

        key = _unique_key(base_key, used_keys)
        used_keys.add(key)
        result.append({"key": key, "label": label, "type": ftype})
    return result


def extract_template(
    img_path: str | Path,
    config: "TemplateConfig | None" = None,
    document_type: str = "generic",
    country: str | None = None,
) -> dict:
    """
    Identifica los campos de un documento a partir de su imagen
    Devuelve {personal: [...], document: [...]} con {key, label, type}
    """
    if config is None:
        config = TemplateConfig()

    print(f"  [TemplateOCR] Enviando imagen a {config.ollama_model} (timeout={config.ollama_timeout}s)...")
    raw = _call_llm(img_path, config.ollama_url, config.ollama_model, document_type, timeout=config.ollama_timeout)

    if not raw:
        print("  [TemplateOCR] Warning: el modelo no devolvió campos")
        return {
            "personal":    [],
            "document":    [],
            "fingerprint": {},
            "validators":  {},
            "n_fields":    0,
        }

    used_keys: set = set()
    personal  = _normalize_category(raw.get("personal", []), "personal", used_keys)
    document  = _normalize_category(raw.get("document", []), "document", used_keys)

    fingerprint_raw = raw.get("fingerprint") or {}
    fingerprint = {
        "layout_desc": str(fingerprint_raw.get("layout_desc") or "").strip() or None,
        "anchors":     [str(a).strip() for a in (fingerprint_raw.get("anchors") or []) if str(a).strip()],
    }

    validators_raw = raw.get("validators") or {}
    all_keys       = {f["key"] for f in personal} | {f["key"] for f in document}
    validators     = (
        {k: v for k, v in validators_raw.items() if k in all_keys and isinstance(v, dict)}
        if isinstance(validators_raw, dict) else {}
    )

    n = len(personal) + len(document)
    print(
        f"  [TemplateOCR] {n} campos identificados ({len(personal)} personal, "
        f"{len(document)} document), fingerprint.anchors={len(fingerprint['anchors'])}, "
        f"validators={len(validators)}"
    )
    return {
        "personal":    personal,
        "document":    document,
        "fingerprint": fingerprint,
        "validators":  validators,
        "n_fields":    n,
    }