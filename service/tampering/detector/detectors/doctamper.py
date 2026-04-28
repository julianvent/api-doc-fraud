"""DocTamper adapter: turns the low-level engine output into a contract result. """
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..engine import TamperingEngine
from ..report import DocTamperResult, FaceDetection, Region, Zone
from ..thresholds import Thresholds


class DocTamperDetector:
    name = "doctamper"

    def __init__(self, engine: TamperingEngine) -> None:
        self._engine = engine

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def analyze(
        self,
        image_rgb: np.ndarray,
        face: FaceDetection,
        thresholds: Thresholds,
    ) -> Tuple[DocTamperResult, np.ndarray]:
        """Return (result, heatmap). The heatmap is the raw per-pixel map used
        for artifact rendering; it never appears in the contract itself."""
        heatmap, score_mean = self._engine.detect(image_rgb)
        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

        score_outside = _mean_outside_face(heatmap, face)
        regions = _extract_regions(heatmap, face, thresholds)

        result = DocTamperResult(
            ran=True,
            score_mean=round(float(score_mean), 4),
            score_max=round(float(heatmap.max()) if heatmap.size else 0.0, 4),
            score_outside_face=round(score_outside, 4),
            regions=regions,
        )
        return result, heatmap


def _mean_outside_face(heatmap: np.ndarray, face: FaceDetection) -> float:
    if not face.detected or face.bbox is None:
        return float(heatmap.mean()) if heatmap.size else 0.0

    h, w = heatmap.shape[:2]
    x, y, bw, bh = face.bbox
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x + bw))
    y2 = max(0, min(h, y + bh))
    if x2 <= x1 or y2 <= y1:
        return float(heatmap.mean()) if heatmap.size else 0.0

    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    outside = heatmap[mask]
    if outside.size == 0:
        return float(heatmap.mean())
    return float(outside.mean())


def _extract_regions(
    heatmap: np.ndarray, face: FaceDetection, thresholds: Thresholds,
) -> List[Region]:
    h, w = heatmap.shape[:2]
    total_area = h * w
    min_area_px = max(1, int(thresholds.min_region_area_fraction * total_area))

    binary = (heatmap >= thresholds.binary_threshold).astype(np.uint8) * 255
    if binary.sum() == 0:
        return []

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    face_norm = _normalized_face_bbox(face, (h, w))
    regions: List[Region] = []

    for i in range(1, num_components):
        x, y, cw, ch, area = stats[i]
        if area < min_area_px:
            continue
        component_mask = (labels == i)
        local_score = float(heatmap[component_mask].mean())
        bbox = (
            round(x / w, 4),
            round(y / h, 4),
            round((x + cw) / w, 4),
            round((y + ch) / h, 4),
        )
        zone = _classify_zone(bbox, face_norm)
        label = (
            "edited text region" if zone == Zone.TEXT
            else "anomaly overlapping face region"
        )
        regions.append(Region(
            detector="doctamper",
            bbox=bbox,
            area_fraction=round(area / total_area, 4),
            score=round(local_score, 3),
            zone=zone,
            label=label,
        ))

    regions.sort(key=lambda r: r.score, reverse=True)
    return regions


def _normalized_face_bbox(
    face: FaceDetection, shape: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    if not face.detected or face.bbox is None:
        return None
    h, w = shape
    x, y, bw, bh = face.bbox
    return (x / w, y / h, (x + bw) / w, (y + bh) / h)


def _classify_zone(
    region_bbox: Tuple[float, float, float, float],
    face_bbox_norm: Optional[Tuple[float, float, float, float]],
) -> Zone:
    if face_bbox_norm is None:
        return Zone.TEXT
    rx1, ry1, rx2, ry2 = region_bbox
    fx1, fy1, fx2, fy2 = face_bbox_norm
    # Use the region centroid so partial overlap at the face border does not
    # force a PHOTO classification on regions that live mostly in the text.
    cx = (rx1 + rx2) / 2
    cy = (ry1 + ry2) / 2
    if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
        return Zone.PHOTO
    return Zone.TEXT
