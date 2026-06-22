"""Image overlays for debug + stakeholder demos.

Pure NumPy/OpenCV. Overlays are drawn on a copy of the image; callers
receive the annotated array and persist it as they like.
"""

from __future__ import annotations

import cv2
import numpy as np

from service.liveness.domain.report import LivenessReport
from service.liveness.domain.verdict import Verdict


# BGR colors per verdict (OpenCV ordering).
_VERDICT_COLORS: dict[Verdict, tuple[int, int, int]] = {
    Verdict.ACCEPT: (60, 180, 60),       # green
    Verdict.REVIEW: (40, 180, 220),      # amber
    Verdict.HARD_REJECT: (40, 50, 220),  # red
}

_LANDMARK_COLOR = (240, 240, 240)
_TEXT_COLOR = (255, 255, 255)
_PANEL_BG = (24, 24, 24)


def annotate(image: np.ndarray, report: LivenessReport) -> np.ndarray:
    """Return an annotated copy of `image` with the report overlay.

    Shows a coloured face rectangle (if any), landmark dots, a top
    banner with verdict + score, and a right-side panel of per-detector
    scores and reasons.
    """
    annotated = image.copy()
    color = _VERDICT_COLORS.get(report.verdict, (200, 200, 200))

    if report.face_crop is not None:
        bb = report.face_crop.bbox
        cv2.rectangle(annotated, (bb.x, bb.y), (bb.x_max, bb.y_max), color, 2)
        for lm in report.face_crop.landmarks:
            cv2.circle(annotated, (int(lm.x), int(lm.y)), 3, _LANDMARK_COLOR, -1)

    _draw_top_banner(annotated, report, color)
    _draw_side_panel(annotated, report)
    return annotated


def _draw_top_banner(canvas: np.ndarray, report: LivenessReport, color) -> None:
    h, w = canvas.shape[:2]
    banner_h = max(56, h // 12)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _PANEL_BG, thickness=-1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, dst=canvas)

    title = f"{report.verdict.value}"
    score = f"score={report.score:.3f}"
    cv2.putText(
        canvas, title, (16, banner_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, score, (320, banner_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _TEXT_COLOR, 2, cv2.LINE_AA,
    )


def _draw_side_panel(canvas: np.ndarray, report: LivenessReport) -> None:
    if not report.detectors and not report.reasons:
        return
    h, w = canvas.shape[:2]
    panel_w = max(220, w // 4)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), _PANEL_BG, thickness=-1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, dst=canvas)

    line_h = 22
    y = 80
    x = w - panel_w + 14

    cv2.putText(
        canvas, "Detectors", (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOR, 1, cv2.LINE_AA,
    )
    y += line_h
    for det_name, det in report.detectors.items():
        text = f"{det_name}: {det.score:.2f} [{det.status.value}]"
        cv2.putText(
            canvas, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
        y += line_h

    if report.reasons:
        y += line_h // 2
        cv2.putText(
            canvas, "Reasons", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
        y += line_h
        for reason in report.reasons[:8]:
            cv2.putText(
                canvas, f"- {reason}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT_COLOR, 1, cv2.LINE_AA,
            )
            y += line_h
