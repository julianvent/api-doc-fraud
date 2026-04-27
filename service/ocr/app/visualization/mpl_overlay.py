from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from ..models import OCRResult, Word

_EN_COLOR = "#22c55e"
_HI_COLOR = "#f97316"
_LOW_COLOR = "#ef4444"

_font_registered = False


def _register_font(font_path: str) -> None:
    """Register the Devanagari font once per process so labels render."""
    global _font_registered
    if _font_registered:
        return
    path = Path(font_path)
    if path.is_file():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = "Noto Sans Devanagari"
    _font_registered = True


def render(
    image: np.ndarray,
    ocr: OCRResult,
    output_path: str | Path,
    font_path: str,
) -> Path:
    """Write a PNG with matplotlib-rendered bboxes + Devanagari labels."""
    _register_font(font_path)

    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_rgb.shape[:2]
    for word in ocr.english.words:
        _draw(ax, word, w, h, _EN_COLOR, label=True)
    for word in ocr.hindi.words:
        _draw(ax, word, w, h, _HI_COLOR, label=True)
    for word in ocr.low_confidence:
        _draw(ax, word, w, h, _LOW_COLOR, label=False)

    legend = [
        patches.Patch(edgecolor=_EN_COLOR, facecolor="none", label="English"),
        patches.Patch(edgecolor=_HI_COLOR, facecolor="none", label="Hindi"),
        patches.Patch(edgecolor=_LOW_COLOR, facecolor="none", label="Low confidence"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw(ax, word: Word, w: int, h: int, color: str, label: bool) -> None:
    (x1, y1), (x2, y2) = word.geometry
    rect = patches.Rectangle(
        (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
        linewidth=1.5, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1 * w, y1 * h - 4, word.text,
                fontsize=6, color=color, va="bottom")
