import json
import os
import re
import requests
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageOps

from .model import TemplateTextLine, TemplateConfig
from .prompt import _PARSE_PROMPT, _build_prompt


_TEXT_CATEGORIES = {
    "Text", "Title", "Section-header", "List-item",
    "Caption", "Footnote", "Table",
}

Y_PADDING_PX   = 4
MAX_GAP_RATIO  = 0.15


def _load_image(path: str | Path, max_side: int = 1_600) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return np.array(img)


class _DotsWrapper:

    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model_path = os.path.abspath(model_path)

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        print(f"  [TemplateOCR/dots] Model : {self._model_path}")
        print(f"  [TemplateOCR/dots] Device: {self._device}")

        attn_impl = "flash_attention_2" if self._device == "cuda" else "sdpa"
        dtype     = __import__("torch").float32 if self._device == "cpu" else __import__("torch").bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self._device != "cuda":
            self._model = self._model.to(self._device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.std(r.astype(int) - g.astype(int)) < 3:
            pil = PILImage.fromarray(image)
            return np.array(ImageOps.autocontrast(pil, cutoff=2))
        return image

    def run(self, image: np.ndarray) -> list[TemplateTextLine]:
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        image     = self._preprocess(image)
        pil_image = PILImage.fromarray(image)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": _PARSE_PROMPT},
        ]}]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._device)

        inputs.pop("mm_token_type_ids", None)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=8_000)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return self._parse(raw, image.shape)

    def _parse(self, raw: str, image_shape: tuple) -> list[TemplateTextLine]:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []

        elements = data if isinstance(data, list) else data.get("elements", data.get("layout", []))
        h, w     = image_shape[:2]
        lines    = []

        for elem in elements:
            category = elem.get("category", elem.get("type", "Text"))
            text     = elem.get("text", elem.get("content", "")).strip()
            bbox_raw = elem.get("bbox", [])

            if category not in _TEXT_CATEGORIES or not text:
                continue

            if len(bbox_raw) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox_raw)
                bbox = np.array(
                    [[max(0,x1), max(0,y1)], [min(w,x2), max(0,y1)],
                     [min(w,x2), min(h,y2)], [max(0,x1), min(h,y2)]],
                    dtype=np.int32,
                )
            else:
                bbox = np.zeros((4, 2), dtype=np.int32)

            lines.append(TemplateTextLine(text=text, bbox=bbox, category=category))

        return lines

_MRZ_PATTERN = re.compile(r"^[A-Z0-9<]{15,}$")

def _is_mrz(text: str) -> bool:
    """MRZ lines son largas, solo mayúsculas, números y '<'. Nunca son fields."""
    lines = text.replace("\n", " ").split()
    # Si la mayoría del texto son secuencias tipo MRZ, entonces es MRZ
    mrz_chars = sum(1 for c in text if c.isupper() or c.isdigit() or c == "<")
    return mrz_chars / max(len(text), 1) > 0.85 and len(text) > 20

# LLM pairer
def _call_llm(
    elements: list[TemplateTextLine],
    ollama_url: str,
    ollama_model: str,
    img_w: int,
    img_h: int,
    document_type: str = "generic",
    explicit: bool = False
) -> tuple[list[dict], list[dict]]:
    elements_text = "\n".join(
        f"{i}: {repr(e.text)} "
        f"[pos: top={round(e.bbox[:,1].min()/img_h, 2)}, "
        f"left={round(e.bbox[:,0].min()/img_w, 2)}, "
        f"bottom={round(e.bbox[:,1].max()/img_h, 2)}, "
        f"right={round(e.bbox[:,0].max()/img_w, 2)}]"
        for i, e in enumerate(elements)
    )
    prompt = _build_prompt(elements_text, document_type, explicit=explicit)

    try:
        response = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
    except Exception as e:
        print(f"  [TemplateOCR/LLM] Error: {e}")
        return [], [] # para anchors

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
                return [], [] # para anchors
        else:
            return [], [] # para anchors

    if not isinstance(data, dict):
        return [], [] # para anchors

    fields  = data.get("fields",  [])
    anchors = data.get("anchors", [])

    #print(f"  [TemplateOCR/LLM] {len(anchors)} anchors ignorados: "
    #      f"{[a.get('text', '')[:30] for a in anchors]}")

    return fields, anchors


# Bbox helpers
def _bbox_to_xyxy(bbox: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(bbox[:, 0].min()), int(bbox[:, 1].min()),
        int(bbox[:, 0].max()), int(bbox[:, 1].max()),
    )


def _normalize(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> dict:
    return {
        "x1": round(max(x1, 0) / img_w, 4),
        "y1": round(max(y1, 0) / img_h, 4),
        "x2": round(min(x2, img_w) / img_w, 4),
        "y2": round(min(y2, img_h) / img_h, 4),
    }


def _split_bbox_vertically(bbox: np.ndarray) -> tuple[tuple, tuple]:
    """Divide el bbox por la mitad. Se usa cuando label y value están en el mismo elemento."""
    x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
    mid = (y1 + y2) // 2
    return (x1, y1, x2, mid), (x1, mid, x2, y2)


def _estimate_value_region(
    label_xyxy: tuple[int, int, int, int],
    all_boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    #expand_x: bool = True,
) -> dict:
    """Busca el vecino más cercano (derecha o abajo). Si no hay, estima debajo del label."""
    lx1, ly1, lx2, ly2 = label_xyxy
    lh        = ly2 - ly1
    max_gap_x = img_w * MAX_GAP_RATIO
    max_gap_y = img_h * MAX_GAP_RATIO

    best_dist, best_box, best_dir = float("inf"), None, None

    for cx1, cy1, cx2, cy2 in all_boxes:
        if cx1 > lx2 and not (ly2 < cy1 or ly1 > cy2):
            gap = cx1 - lx2
            if gap <= max_gap_x and gap < best_dist:
                best_dist, best_box, best_dir = gap, (cx1, cy1, cx2, cy2), "right"
        if cy1 > ly2 and not (lx2 < cx1 or lx1 > cx2):
            gap = cy1 - ly2
            if gap <= max_gap_y and gap < best_dist:
                best_dist, best_box, best_dir = gap, (cx1, cy1, cx2, cy2), "below"

    if best_dir == "right":
        _, cy1, _, cy2 = best_box
        return {
            "x1": round(lx2 / img_w + 0.005, 4),
            "y1": round(max(min(ly1, cy1) - Y_PADDING_PX, 0) / img_h, 4),
            "x2": round(cx2 / img_w, 4),
            "y2": round(min(max(ly2, cy2) + Y_PADDING_PX, img_h) / img_h, 4),
        }
    if best_dir == "below":
        cx1, cy1, cx2, cy2 = best_box
        return {
            "x1": round(min(lx1, cx1) / img_w, 4),
            "y1": round(max(cy1 - Y_PADDING_PX, 0) / img_h, 4),
            "x2": round(cx2 / img_w, 4),
            "y2": round(min(cy2 + Y_PADDING_PX, img_h) / img_h, 4),
        }

    return {
        "x1": round(lx1 / img_w, 4),
        "y1": round(max(ly2 - Y_PADDING_PX, 0) / img_h, 4),
        "x2": round(lx2 / img_w, 4),
        "y2": round(min(ly2 + max(lh, 20) + Y_PADDING_PX, img_h) / img_h, 4),
    }


# Slug
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


# Builder
def _build_fields(
    elements: list[TemplateTextLine],
    pairs: list[dict],
    img_w: int,
    img_h: int,
    anchor_indices: set = None,
    #expand_x: bool = True,
) -> list[dict]:
    all_boxes = [_bbox_to_xyxy(e.bbox) for e in elements]
    used_keys : set       = set()
    fields    : list[dict] = []
    anchor_indices = anchor_indices or set()

    for pair in pairs:
        label_text = str(pair.get("label_text", "")).strip()
        value_text = str(pair.get("value_text", "")).strip()
        label_idx  = pair.get("label_idx")
        value_idx  = pair.get("value_idx")

        if not label_text or label_idx is None:
            continue
        if not (0 <= label_idx < len(elements)):
            continue

        label_bbox = elements[label_idx].bbox
        label_xyxy = _bbox_to_xyxy(label_bbox)

        if value_idx == label_idx:
            # Label y value en el mismo bloque — dividir el bbox verticalmente
            top, bottom = _split_bbox_vertically(label_bbox)
            label_region = _normalize(*top, img_w=img_w, img_h=img_h)
            value_region = {
                "x1": round(bottom[0] / img_w, 4),
                "y1": round(max(bottom[1] - Y_PADDING_PX, 0) / img_h, 4),
                "x2": round(bottom[2] / img_w, 4),
                "y2": round(min(bottom[3] + Y_PADDING_PX, img_h) / img_h, 4),
            }

        elif value_idx is not None and 0 <= value_idx < len(elements):
            # Label y value en elementos distintos
            if value_idx in anchor_indices:
                label_region = _normalize(*label_xyxy, img_w=img_w, img_h=img_h)
                other_boxes  = [b for j, b in enumerate(all_boxes) if j != label_idx and j not in anchor_indices]
                value_region = _estimate_value_region(label_xyxy, other_boxes, img_w, img_h)
            else:
                vx1, vy1, vx2, vy2 = _bbox_to_xyxy(elements[value_idx].bbox)
                label_region = _normalize(*label_xyxy, img_w=img_w, img_h=img_h)
                value_region = {
                    "x1": round(min(label_xyxy[0], vx1) / img_w, 4),
                    "y1": round(max(vy1 - Y_PADDING_PX, 0) / img_h, 4),
                    "x2": round(vx2 / img_w, 4),
                    "y2": round(min(vy2 + Y_PADDING_PX, img_h) / img_h, 4),
                }

        else:
            # LLM no dio value_idx — fallback espacial
            label_region = _normalize(*label_xyxy, img_w=img_w, img_h=img_h)
            other_boxes  = [b for j, b in enumerate(all_boxes) if j != label_idx]
            value_region = _estimate_value_region(label_xyxy, other_boxes, img_w, img_h) #, expand_x=expand_x)

        key = _unique_key(_slugify(label_text), used_keys)
        used_keys.add(key)

        fields.append({
            "key":          key,
            "label":        label_text,
            "label_region": label_region,
            "value_region": value_region,
        })

    return fields


# Singleton

_dots_instance: _DotsWrapper | None = None

def _get_dots(model_path: str) -> _DotsWrapper:
    global _dots_instance
    if _dots_instance is None:
        _dots_instance = _DotsWrapper(model_path)
    return _dots_instance


def _merge_bilingual_fields(fields: list[dict]) -> list[dict]:
    """
    Fusiona campos duplicados que son el mismo label en dos idiomas.
    Detecta duplicados por value_region idéntico o muy similar.
    """
    merged  : list[dict] = []
    used_idx: set        = set()

    for i, f in enumerate(fields):
        if i in used_idx:
            continue

        # Busca otro field con el mismo value_region (mismo valor, label distinto)
        duplicate = None
        for j, g in enumerate(fields):
            if j <= i or j in used_idx:
                continue
            # Mismo value_region → mismo campo en otro idioma
            if f["value_region"] == g["value_region"]:
                duplicate = j
                break

        if duplicate is not None:
            g = fields[duplicate]
            # Fusionar: combinar los dos labels en uno bilingüe
            merged.append({
                "key":          f["key"],
                "label":        f"{f['label']} / {g['label']}",
                "label_region": f["label_region"],  # region del primer label
                "value_region": f["value_region"],
            })
            used_idx.add(i)
            used_idx.add(duplicate)
        else:
            merged.append(f)
            used_idx.add(i)

    return merged


def extract_template(
    img_path: str | Path,
    config: "TemplateConfig | None" = None,
    #expand_x: bool = True,
    document_type: str = "generic",
    explicit: bool = False,
    include_type: bool = False,
) -> dict:
    """
    Genera el template de campos de un documento a partir de una imagen
    de referencia limpia.

    Devuelve:
        {
            "fields": [
                {
                    "key":          str,
                    "label":        str,
                    "label_region": {"x1", "y1", "x2", "y2"},
                    "value_region": {"x1", "y1", "x2", "y2"},
                },
                ...
            ],
            "n_fields": int,
            "engine":   "dots+llm"
        }
    """
    if config is None:
        config = TemplateConfig()

    dots  = _get_dots(config.dots_model_path)
    image = _load_image(Path(img_path))
    img_h, img_w = image.shape[:2]

    print(f"  [TemplateOCR] Paso 1/2 — dots.ocr extrayendo elementos...")
    elements = dots.run(image)
    print(f"  [TemplateOCR] {len(elements)} elementos detectados")

    if not elements:
        return {"fields": [], "n_fields": 0, "engine": "dots+llm"}

    elements_filtered = [e for e in elements if not _is_mrz(e.text)]
    print(f"  [TemplateOCR] {len(elements) - len(elements_filtered)} MRZ lines excluidas")
    
    print(f"  [TemplateOCR] Paso 2/2 — LLM identificando label-value pairs...")
    pairs, anchors = _call_llm(
        elements_filtered, 
        config.ollama_url,
        config.ollama_model, 
        img_w, img_h, 
        document_type=document_type,
        explicit=explicit) # Aqui se pueden devolver los anchors
    print(f"  [TemplateOCR] {len(pairs)} pares identificados")

    if not pairs:
        print("  [TemplateOCR] Warning: LLM no devolvió pares")
        return {"fields": [], "n_fields": 0, "anchors": [], 
                "engine": "dots+llm"}

    anchor_indices = {a.get("idx") for a in anchors if "idx" in a}

    fields = _build_fields(elements_filtered, pairs, img_w, img_h, anchor_indices) #, expand_x=expand_x)
    if  document_type.lower() == "passport":
        fields = _merge_bilingual_fields(fields)

    return {
        "fields":   fields,
        "n_fields": len(fields),
        #"anchors":  anchors,
        "engine":   "dots+llm",
    }