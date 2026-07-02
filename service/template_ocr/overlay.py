"""Debug utility: draw DetectedElement boxes or template field regions over an image.

draw_element_boxes  — dibuja polígonos de DetectedElement (gris).
draw_template_regions — dibuja label_region (azul) y value_region (verde)
                        de cada FieldSpec del template guardado.

Precondición de bbox: todos los elementos deben tener bbox normalizado 0–1
(igual que la precondición de textlines_to_elements). Si algún bbox contiene
valores > 1.5 la función lanza ValueError antes de dibujar nada — igual que
textlines_to_elements, para no dibujar basura en silencio si entran píxeles
absolutos.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from service.template_ocr.elements import DetectedElement


def draw_element_boxes(
    img: np.ndarray,
    elements: list[DetectedElement] | list[dict],
    color: tuple[int, int, int] = (80, 80, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of img with each element's polygon bbox drawn in gray.

    img      : RGB or grayscale uint8 array, shape (H, W) or (H, W, 3).
    elements : list of DetectedElement objects OR list of dicts from to_dict().
    color    : BGR color for the outline. Default is dark gray.

    Precondición: cada bbox debe estar normalizado 0–1. Se valida antes de
    dibujar: si cualquier bbox tiene max > 1.5, se lanza ValueError. El umbral
    1.5 es el mismo que en textlines_to_elements — permite ruido de float pero
    rechaza píxeles absolutos (que serían decenas o cientos).

    Raises:
        ValueError: si algún bbox no está normalizado.
    """
    # --- validate all bboxes before touching the image ---
    for i, el in enumerate(elements):
        bbox = el.bbox if isinstance(el, DetectedElement) else np.array(el["bbox"], dtype=np.float32)
        if bbox.size > 0 and float(bbox.max()) > 1.5:
            raise ValueError(
                f"Element {i} (id={el.id if isinstance(el, DetectedElement) else el.get('id')}) "
                f"has non-normalized bbox (max={float(bbox.max()):.1f}). "
                "Expected values in [0, 1]. Pass normalized bboxes (Paddle path)."
            )

    # --- convert to BGR for drawing ---
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

    h, w = out.shape[:2]

    for el in elements:
        if isinstance(el, DetectedElement):
            bbox = el.bbox
            label = str(el.id)
        else:
            bbox = np.array(el["bbox"], dtype=np.float32)
            label = str(el.get("id", "?"))

        pts = (bbox * np.array([w, h])).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        x0 = int(pts[:, 0].min())
        y0 = int(pts[:, 1].min())
        cv2.putText(
            out, label, (x0, max(y0 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_template_regions(
    img: np.ndarray,
    fields: list[Any],
    label_color: tuple[int, int, int] = (200, 100, 0),
    value_color: tuple[int, int, int] = (0, 180, 60),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of img with each field's label/value regions drawn.

    img    : RGB uint8 array (H, W, 3) or grayscale (H, W).
    fields : list of FieldSpec objects or dicts with 'key', 'label_region',
             'value_region'. Regions are {x1, y1, x2, y2} dicts normalized
             0–1. Fields without regions are silently skipped.
    label_color : BGR color for label_region rectangles (default: blue).
    value_color : BGR color for value_region rectangles (default: green).
    """
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

    h, w = out.shape[:2]

    def _draw_rect(region: dict, color: tuple, tag: str, key: str) -> None:
        x1 = int(region["x1"] * w)
        y1 = int(region["y1"] * h)
        x2 = int(region["x2"] * w)
        y2 = int(region["y2"] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out, f"{tag}:{key}", (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )

    for f in fields:
        if hasattr(f, "key"):
            key, label_region, value_region = f.key, f.label_region, f.value_region
        else:
            key = f.get("key", "?")
            label_region = f.get("label_region")
            value_region = f.get("value_region")

        if label_region:
            _draw_rect(label_region, label_color, "L", key)
        if value_region:
            _draw_rect(value_region, value_color, "V", key)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
