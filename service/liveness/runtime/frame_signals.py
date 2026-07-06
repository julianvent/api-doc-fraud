"""Per-frame inference result fed into an ActiveSession.

`FrameSignals` is the contract between `InferenceEngine` and
`ActiveSession.submit()` — exactly the kwargs `submit()` expects,
bundled together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from service.liveness.active.occlusion import FaceOcclusionReport
from service.liveness.domain.face import FaceCrop
from service.liveness.runtime.config import OvalGeometry


@dataclass(frozen=True, slots=True)
class FrameSignals:
    """Everything `ActiveSession.submit()` needs for one frame.

    `occlusion`/`embedding` are `None` when the caller asked the engine
    to skip them this frame (parser throttle / non-neutral phase).
    """

    face: FaceCrop | None
    in_oval: bool
    eye_blink: float | None
    mouth_smile: float | None
    occlusion: FaceOcclusionReport | None
    embedding: np.ndarray | None


def compute_in_oval(
    face: FaceCrop | None,
    frame_w: int,
    frame_h: int,
    oval: OvalGeometry,
) -> bool:
    """True when the face is centered inside the guide oval at a good size.

    Mirrors the demo's `_check_oval`: the bbox center must lie inside the
    ellipse and the bbox width fall within the face-width band.
    Resolution-independent — `oval` is in fractions.
    """
    if face is None:
        return False
    bb = face.bbox
    face_cx = bb.x + bb.width / 2.0
    face_cy = bb.y + bb.height / 2.0
    cx, cy = oval.cx * frame_w, oval.cy * frame_h
    sw, sh = oval.rx * frame_w, oval.ry * frame_h
    if sw <= 0 or sh <= 0:
        return False
    norm_dist = ((face_cx - cx) / sw) ** 2 + ((face_cy - cy) / sh) ** 2
    if norm_dist >= 1.0:
        return False
    oval_w = sw * 2.0
    ratio = bb.width / oval_w if oval_w > 0 else 0.0
    return oval.face_width_ratio_min <= ratio <= oval.face_width_ratio_max
