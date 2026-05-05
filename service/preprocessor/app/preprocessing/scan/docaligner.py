"""ONNX inference wrapper for DocAligner heatmap-regression models.

The model emits a (1, 4, H, W) tensor where each channel is a heatmap for one
corner. We binarise each heatmap, find the largest blob, and use its centroid
as the corner coordinate (rescaled back to the original image space).

The ONNX session is loaded once and cached as a module-level singleton — the
first call pays ~300 ms; subsequent calls only pay inference cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from ...config import ScanConfig

_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"

_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None


def _get_session(model_filename: str) -> ort.InferenceSession:
    global _session, _input_name
    if _session is None:
        model_path = _WEIGHTS_DIR / model_filename
        if not model_path.is_file():
            raise FileNotFoundError(
                f"DocAligner weights not found at {model_path}. "
                f"Download from the DocAligner Google Drive folder and place the file there."
            )
        _session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
    return _session


def predict_corners(image: np.ndarray, cfg: ScanConfig) -> Optional[np.ndarray]:
    """Run DocAligner on a BGR image and return 4 corners in original-image space.

    Returns None if any corner heatmap is empty after thresholding.
    """
    session = _get_session(cfg.model_filename)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (cfg.model_input_size, cfg.model_input_size))
    tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, ...]

    outputs = session.run(None, {_input_name: tensor})
    heatmaps = outputs[0][0]  # (4, H, W)

    h, w = image.shape[:2]
    corners: list[list[float]] = []
    for i in range(4):
        point = _heatmap_to_point(heatmaps[i], (w, h), cfg.heatmap_threshold)
        if point is None:
            return None
        corners.append(point)

    return np.array(corners, dtype=np.float32)


def _heatmap_to_point(
    heatmap: np.ndarray, target_size: tuple[int, int], threshold: float
) -> Optional[list[float]]:
    """Resize heatmap, threshold, and return centroid of the largest blob."""
    resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    mask = (resized >= threshold).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return [float(cx), float(cy)]
