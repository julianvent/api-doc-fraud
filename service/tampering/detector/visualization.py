"""Rendering utilities for tampering detection outputs.

Artifacts produced per page:
  * heatmap PNG: raw 8-bit grayscale of the model probability map, for
    archival and manual inspection in image viewers.
  * overlay PNG: the page image blended with a JET-colormapped heatmap,
    annotated with suspicious-region bboxes, the face localizer bbox, and a
    top-left banner summarizing the verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .report import FaceDetection, Region, Verdict, Zone

_HEATMAP_COLORMAP = cv2.COLORMAP_JET

_VERDICT_BANNER_BGR = {
    Verdict.ACCEPT: (60, 140, 40),
    Verdict.REVIEW: (0, 165, 255),
    Verdict.HARD_REJECT: (40, 40, 200),
}

_ZONE_BBOX_BGR = {
    Zone.TEXT: (0, 255, 0),
    Zone.PHOTO: (0, 165, 255),
    Zone.TEMPLATE: (255, 200, 0),
    Zone.UNKNOWN: (200, 200, 200),
}

_FACE_BBOX_BGR = (255, 0, 255)
_LABEL_TEXT_BGR = (255, 255, 255)
_LABEL_BG_BGR = (0, 0, 0)
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.6
_LABEL_THICKNESS = 1
_LABEL_PADDING = 4


def save_heatmap(heatmap: np.ndarray, output_path: Path) -> Path:
    """Persist a raw [0, 1] heatmap as an 8-bit grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(output_path), arr)
    return output_path


def save_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    regions: Sequence[Region],
    output_path: Path,
    face: Optional[FaceDetection] = None,
    verdict: Optional[Verdict] = None,
    verdict_score: Optional[float] = None,
    alpha: float = 0.5,
) -> Path:
    """Render a human-readable overlay and persist it as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, _HEATMAP_COLORMAP)
    composite = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0.0)

    h, w = composite.shape[:2]
    bbox_thickness = max(2, min(h, w) // 500)

    if face is not None and face.detected and face.bbox is not None:
        fx, fy, fw, fh = face.bbox
        cv2.rectangle(
            composite, (fx, fy), (fx + fw, fy + fh),
            _FACE_BBOX_BGR, bbox_thickness,
        )
        _draw_label(composite, f"face {face.confidence:.2f}", (fx, fy))

    for idx, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region.bbox
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        color = _ZONE_BBOX_BGR.get(region.zone, _ZONE_BBOX_BGR[Zone.UNKNOWN])
        cv2.rectangle(composite, pt1, pt2, color, bbox_thickness)
        _draw_label(
            composite,
            f"{idx}:{region.zone.value} {region.score:.2f}",
            pt1,
        )

    if verdict is not None:
        _draw_verdict_banner(composite, verdict, verdict_score)

    cv2.imwrite(str(output_path), composite)
    return output_path


def save_face_crop(
    image_rgb: np.ndarray, face: FaceDetection, output_path: Path,
) -> Optional[Path]:
    """Persist the cropped face region, or return None if no face."""
    if not face.detected or face.bbox is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = face.bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    crop_bgr = cv2.cvtColor(image_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), crop_bgr)
    return output_path


def _draw_verdict_banner(
    image_bgr: np.ndarray, verdict: Verdict, score: Optional[float],
) -> None:
    color = _VERDICT_BANNER_BGR[verdict]
    text = verdict.value if score is None else f"{verdict.value}  score={score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, _LABEL_FONT, 0.9, 2)
    pad = 10
    cv2.rectangle(
        image_bgr, (0, 0), (tw + pad * 2, th + baseline + pad * 2),
        color, thickness=-1,
    )
    cv2.putText(
        image_bgr, text, (pad, th + pad),
        _LABEL_FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_label(image_bgr: np.ndarray, text: str, anchor: Tuple[int, int]) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(
        text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS,
    )
    x, y = anchor
    top_left = (x, max(0, y - text_h - baseline - _LABEL_PADDING * 2))
    bottom_right = (x + text_w + _LABEL_PADDING * 2, y)
    cv2.rectangle(image_bgr, top_left, bottom_right, _LABEL_BG_BGR, thickness=-1)
    cv2.putText(
        image_bgr, text,
        (x + _LABEL_PADDING, y - baseline - _LABEL_PADDING // 2),
        _LABEL_FONT, _LABEL_SCALE, _LABEL_TEXT_BGR, _LABEL_THICKNESS,
        cv2.LINE_AA,
    )
