from abc import ABC, abstractmethod
import os
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from PIL import Image as PILImage


from service.logging_config import get_logger
from .models import Config, TextLine
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

import numpy as np


log = get_logger(__name__)


def load_image(path: str | Path, max_side: int = 1_600) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        log.debug("resized image: %dx%d → %dx%d", w, h, new_w, new_h)
    return np.array(img)

def _autocontrast_if_grayscale(image: np.ndarray) -> np.ndarray:
    """Apply autocontrast when the image appears grayscale (R≈G≈B channels).
    Improves detection in dark or low-contrast regions."""
    from PIL import ImageOps
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if np.std(r.astype(int) - g.astype(int)) < 3:
        pil = PILImage.fromarray(image)
        pil = ImageOps.autocontrast(pil, cutoff=2)
        return np.array(pil)
    return image


class OCREngine(ABC):
    """
    Contract every OCR backend must satisfy.
    Subclass and implement extract() to swap engines without touching
    language classification, KIE, MRZ, visualisation code or agent code.
    """

    @abstractmethod
    def extract(
        self,
        image: "np.ndarray | str | Path",
        max_side: int = 1_600,
    ) -> list[TextLine]:
        """Run OCR on an RGB numpy image (or path). Return one TextLine per region."""
        ...

class PaddleOCRAdapter(OCREngine):
    """PaddleOCR v3.x implementation of OCREngine."""

    def __init__(self, config: Config, **paddle_kwargs) -> None:
        from paddleocr import PaddleOCR
        self._threshold = config.confidence_threshold
        self._engine = PaddleOCR(
            lang="hi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **paddle_kwargs,
        )

    def extract(
        self,
        image: "np.ndarray | str | Path",
        max_side: int = 1_600,
    ) -> list[TextLine]:

        if isinstance(image, (str, Path)):
            image = load_image(image, max_side)

        lines: list[TextLine] = []
        for result in self._engine.predict(image):
            res    = result.get("res", result)
            texts  = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            bboxes = res.get("dt_polys", res.get("rec_polys", res.get("det_polys", [])))
            for text, score, bbox in zip(texts, scores, bboxes):
                if score >= self._threshold and text.strip():
                    h, w = image.shape[:2]

                    bbox_array = np.array(bbox, dtype=np.float32)
                    bbox_array[:, 0] /= w
                    bbox_array[:, 1] /= h

                    lines.append(TextLine(
                        text=text.strip(),
                        confidence=round(float(score), 4),
                        bbox=bbox_array,
                    ))
        return lines


_DOTS_PARSE_PROMPT = """\
Please output the layout information from the document image, \
including each layout element's bbox, its category, and the corresponding \
text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are \
['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', \
'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Content: Extract the complete text within each bbox, \
preserving the original language and script.
4. Reading Order: Elements must be listed in natural reading order.
5. Final Output: The entire output must be a single JSON object.\
"""

# Only these categories contain text relevant for the LLM agent
_DOTS_TEXT_CATEGORIES = {
    "Text", "Title", "Section-header", "List-item",
    "Caption", "Footnote", "Table",
}


class DotsOCRAdapter(OCREngine):

    def __init__(self, config: Config) -> None:
        import torch

        model_path = getattr(config, "dots_model_path", "service/ocr/DotsOCR")
        self._model_path = os.path.abspath(model_path)
        self._threshold  = config.confidence_threshold

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        log.info("[DotsOCR] model=%s device=%s", self._model_path, self._device)

        self._model, self._processor = self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # flash_attention_2 only on CUDA; sdpa is the standard fallback for MPS/CPU
        attn_impl = "flash_attention_2" if self._device == "cuda" else "sdpa"
        dtype     = torch.bfloat16 if self._device != "cpu" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device != "cuda":
            model = model.to(self._device)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )
        return model, processor

    def extract(
        self,
        image: "np.ndarray | str | Path",
        max_side: int = 1_600,
    ) -> list[TextLine]:
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        if isinstance(image, (str, Path)):
            image = load_image(image, max_side)

        image = _autocontrast_if_grayscale(image)
        pil_image = PILImage.fromarray(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": _DOTS_PARSE_PROMPT},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        inputs.pop("mm_token_type_ids", None)
        
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=8_000)

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_output(raw_output, image.shape)

    def _parse_output(self, raw: str, image_shape: tuple) -> list[TextLine]:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print("  [DotsOCR] Warning: output could not be parsed as JSON")
                    return self._fallback_plain_text(cleaned)
            else:
                print("  [DotsOCR] Warning: no JSON found in output")
                return self._fallback_plain_text(cleaned)

        if isinstance(data, list):
            elements = data
        elif isinstance(data, dict):
            elements = data.get("elements", data.get("layout", []))
        else:
            return []

        lines: list[TextLine] = []
        h, w = image_shape[:2]

        for elem in elements:
            category = elem.get("category", elem.get("type", "Text"))
            text     = elem.get("text", elem.get("content", "")).strip()
            bbox_raw = elem.get("bbox", [])

            if category not in _DOTS_TEXT_CATEGORIES or not text:
                continue

            if len(bbox_raw) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox_raw)
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                bbox = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                )
            else:
                bbox = np.zeros((4, 2), dtype=np.int32)

            lines.append(TextLine(text=text, confidence=1.0, bbox=bbox))

        return lines

    def _fallback_plain_text(self, raw: str) -> list[TextLine]:
        """Line-by-line fallback when JSON parsing fails completely."""
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and len(line) > 2:
                lines.append(TextLine(
                    text=line,
                    confidence=0.5,
                    bbox=np.zeros((4, 2), dtype=np.int32),
                ))
        return lines


class DolphinOCRAdapter(OCREngine):

    def __init__(self, config: Config) -> None:
        model_path = getattr(config, "dolphin_model_path", "service/ocr/hf_model")
        repo_path  = getattr(config, "dolphin_repo_path",  "service/ocr/Dolphin")

        self._model_path = os.path.abspath(model_path)
        self._repo_path  = os.path.abspath(repo_path)
        self._threshold  = config.confidence_threshold

        if self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        log.info("[DolphinOCR] model=%s repo=%s", self._model_path, self._repo_path)

        from demo_page import DOLPHIN  # type: ignore  (from the cloned Dolphin repo)
        self._model = DOLPHIN(self._model_path)

    def extract(
        self,
        image: "np.ndarray | str | Path",
        max_side: int = 1_600,
    ) -> list[TextLine]:
        from demo_page import process_single_image  # type: ignore

        if isinstance(image, (str, Path)):
            image = load_image(image, max_side)

        image = _autocontrast_if_grayscale(image)
        pil_image = PILImage.fromarray(image)

        save_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(save_dir, "output_json"),         exist_ok=True)
            os.makedirs(os.path.join(save_dir, "markdown", "figures"), exist_ok=True)

            process_single_image(
                image=pil_image,
                model=self._model,
                save_dir=save_dir,
                image_name="doc",
            )

            json_path = os.path.join(save_dir, "output_json", "doc.json")
            if not os.path.exists(json_path):
                log.warning("[DolphinOCR] no output JSON in %s", save_dir)
                return []

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)

        return self._parse_output(data)

    def _parse_output(self, data: list) -> list[TextLine]:
        lines: list[TextLine] = []

        for elem in data:
            label = elem.get("label", "")
            text  = elem.get("text", "").strip()

            if label == "fig" or not text or text.startswith("!["):
                continue

            try:
                text = text.encode().decode("unicode_escape")
            except Exception:
                pass

            if not text:
                continue

            bbox_raw = elem.get("bbox", [0, 0, 0, 0])
            if len(bbox_raw) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox_raw)
                bbox = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                )
            else:
                bbox = np.zeros((4, 2), dtype=np.int32)

            lines.append(TextLine(text=text, confidence=1.0, bbox=bbox))

        return lines