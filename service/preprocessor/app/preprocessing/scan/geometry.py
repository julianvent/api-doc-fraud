"""Pure geometric helpers for the scanner: ordering, validation, and warping.

These functions never load models or perform I/O — they operate on numpy arrays
and primitive types so they can be tested in isolation.
"""
from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def validate_quad(
    quad: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_corner_angle_deg: float,
    max_corner_angle_deg: float,
) -> bool:
    """Reject degenerate quads: out of bounds, non-convex, wrong size, bad angles."""
    h, w = image_shape[:2]
    frame_area = float(h * w)

    if np.any(np.isnan(quad)) or np.any(np.isinf(quad)):
        return False
    if (quad < 0).any() or (quad[:, 0] > w).any() or (quad[:, 1] > h).any():
        return False

    area = cv2.contourArea(quad.astype(np.float32))
    if area < min_area_ratio * frame_area or area > max_area_ratio * frame_area:
        return False

    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False

    for i in range(4):
        a = quad[(i - 1) % 4]
        b = quad[i]
        c = quad[(i + 1) % 4]
        v1 = a - b
        v2 = c - b
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))
        angle = float(np.degrees(np.arccos(cos_a)))
        if angle < min_corner_angle_deg or angle > max_corner_angle_deg:
            return False

    return True


def four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral region to a fronto-parallel rectangle."""
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)
