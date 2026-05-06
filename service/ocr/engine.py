from abc import ABC, abstractmethod
import os
from pathlib import Path
from PIL import Image as PILImage


from .models import Config, TextLine
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

import numpy as np

def load_image(path: str | Path, max_side: int = 1_600) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        print(f"  Resized: {w}x{h} → {new_w}x{new_h}")
    return np.array(img)

class OCREngine(ABC):
    """
    Contract every OCR backend must satisfy.
    Subclass and implement extract() to swap engines without touching
    language classification, KIE, or visualisation code.
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> list[TextLine]:
        """Run OCR on an RGB numpy image. Return one TextLine per detected region."""
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

    def extract(self,
                image: np.ndarray | str | Path,
                max_side: int = 1_600) -> list[TextLine]:

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
                    lines.append(TextLine(
                        text=text.strip(),
                        confidence=round(float(score), 4),
                        bbox=np.array(bbox, dtype=np.int32),
                    ))
        return lines
