"""Scanner orchestration: model inference → validation → warp or fallback.

This is the only file in the subpackage that knows about both the model and
the geometry helpers. It exposes a single `scan(page, cfg) -> Page`.
"""
from __future__ import annotations

import logging

from ...config import ScanConfig
from ...models import Page
from .docaligner import predict_corners
from .geometry import four_point_transform, order_points, validate_quad

_log = logging.getLogger(__name__)


def scan(page: Page, cfg: ScanConfig) -> Page:
    """Detect the document boundary and warp it to a fronto-parallel view.

    Falls back to the original image (scanned=False) when the model fails or
    the predicted quadrilateral does not pass geometric validation.
    """
    if not cfg.enabled:
        return page

    try:
        corners = predict_corners(page.image, cfg)
    except FileNotFoundError:
        raise
    except Exception as exc:
        _log.warning("DocAligner inference failed for %s: %s", page.source, exc)
        return page

    if corners is None:
        return page

    ordered = order_points(corners)
    if not validate_quad(
        ordered,
        page.image.shape,
        cfg.min_area_ratio,
        cfg.max_area_ratio,
        cfg.min_corner_angle_deg,
        cfg.max_corner_angle_deg,
    ):
        return page

    warped = four_point_transform(page.image, ordered)
    return Page(
        image=warped,
        page_number=page.page_number,
        dpi=page.dpi,
        source=page.source,
        scanned=True,
    )
