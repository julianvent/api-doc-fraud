"""Axis-aligned document localization on raw pixels."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from service.preprocessor.app.config import ScanConfig
from service.preprocessor.app.preprocessing.scan.docaligner import predict_corners
from service.preprocessor.app.preprocessing.scan.geometry import (
    order_points,
    validate_quad,
)
from service.tampering.detector import DocumentLocation

_DEFAULT_CFG = ScanConfig(
    enabled=True,
    model_filename="fastvit_t8_h_e_bifpn_256_fp32.onnx",
    model_input_size=256,
    heatmap_threshold=0.15,
    min_area_ratio=0.20,
    max_area_ratio=0.999,
    min_corner_angle_deg=70.0,
    max_corner_angle_deg=110.0,
)

# Padding fraction around the detected bbox (captures edge watermarks).
_PADDING_FRACTION = 0.08
# Slicing below this size loses more detail than it gains.
_MIN_BBOX_SIDE_PX = 200
# Above this frame fraction, skip the slice (bbox is essentially the frame).
_MAX_BBOX_FRAME_FRACTION = 0.95
# Reject corrupt quads. ID documents sit around 1.4–1.8.
_MAX_ASPECT_RATIO = 3.0


def locate(image_rgb: np.ndarray, cfg: ScanConfig = _DEFAULT_CFG) -> DocumentLocation:
    """Localize the document. Degrades to `localized=False` on failure;
    re-raises only on missing weights (loud startup failure)."""
    h, w = image_rgb.shape[:2]
    frame_shape = (h, w)

    # predict_corners expects BGR.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        corners = predict_corners(image_bgr, cfg)
    except FileNotFoundError:
        raise
    except Exception as exc:
        return DocumentLocation(
            localized=False,
            frame_shape=frame_shape,
            validation_status="MODEL_ERROR",
            fallback_reason=f"docaligner inference raised: {exc.__class__.__name__}",
        )

    if corners is None:
        return DocumentLocation(
            localized=False,
            frame_shape=frame_shape,
            validation_status="NO_CORNERS",
            fallback_reason="docaligner returned no corners",
        )

    ordered = order_points(corners)
    quad_tuple = tuple((float(x), float(y)) for x, y in ordered)

    if not validate_quad(
        ordered, frame_shape,
        cfg.min_area_ratio, cfg.max_area_ratio,
        cfg.min_corner_angle_deg, cfg.max_corner_angle_deg,
    ):
        return DocumentLocation(
            localized=False,
            quad=quad_tuple,
            frame_shape=frame_shape,
            validation_status="INVALID_QUAD",
            fallback_reason="quad failed geometric validation",
        )

    # Axis-aligned bbox from the corners.
    x1 = int(np.floor(ordered[:, 0].min()))
    y1 = int(np.floor(ordered[:, 1].min()))
    x2 = int(np.ceil(ordered[:, 0].max()))
    y2 = int(np.ceil(ordered[:, 1].max()))
    bbox: Tuple[int, int, int, int] = (x1, y1, x2 - x1, y2 - y1)

    # Pad and clamp to frame.
    pad_x = int(round((x2 - x1) * _PADDING_FRACTION))
    pad_y = int(round((y2 - y1) * _PADDING_FRACTION))
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)
    bbox_padded: Tuple[int, int, int, int] = (px1, py1, px2 - px1, py2 - py1)

    bbox_w = px2 - px1
    bbox_h = py2 - py1
    area_fraction = (bbox_w * bbox_h) / float(h * w) if h * w else 0.0

    if bbox_w < _MIN_BBOX_SIDE_PX or bbox_h < _MIN_BBOX_SIDE_PX:
        return DocumentLocation(
            localized=False,
            bbox=bbox,
            bbox_with_padding=bbox_padded,
            quad=quad_tuple,
            frame_shape=frame_shape,
            area_fraction_of_frame=area_fraction,
            validation_status="BBOX_TOO_SMALL",
            fallback_reason=f"bbox side <{_MIN_BBOX_SIDE_PX}px",
        )

    if area_fraction > _MAX_BBOX_FRAME_FRACTION:
        return DocumentLocation(
            localized=False,
            bbox=bbox,
            bbox_with_padding=bbox_padded,
            quad=quad_tuple,
            frame_shape=frame_shape,
            area_fraction_of_frame=area_fraction,
            validation_status="BBOX_COVERS_FRAME",
            fallback_reason="bbox covers full frame; using image as-is",
        )

    # Normalize aspect to >=1 so portrait and landscape share one check.
    raw_aspect = bbox_w / float(bbox_h) if bbox_h else 0.0
    aspect = max(raw_aspect, 1.0 / raw_aspect) if raw_aspect > 0 else 0.0
    if aspect == 0.0 or aspect > _MAX_ASPECT_RATIO:
        return DocumentLocation(
            localized=False,
            bbox=bbox,
            bbox_with_padding=bbox_padded,
            quad=quad_tuple,
            frame_shape=frame_shape,
            area_fraction_of_frame=area_fraction,
            validation_status="INVALID_ASPECT",
            fallback_reason=f"aspect ratio {aspect:.2f} outside expected band",
        )

    return DocumentLocation(
        localized=True,
        bbox=bbox,
        bbox_with_padding=bbox_padded,
        quad=quad_tuple,
        frame_shape=frame_shape,
        area_fraction_of_frame=area_fraction,
        validation_status="OK",
    )
