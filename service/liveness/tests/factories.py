"""Shared fixtures + factories for the liveness test suite.

The state machine is fed pure data (FaceCrop, scalar signals), so the
suite runs without a camera or model — fast and deterministic.
`synthetic_face()` builds a FaceCrop; `feed()` drives a session.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

# Quiet the MediaPipe/TF logs even though the suite doesn't load models.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LIVENESS_LOG_LEVEL", "WARNING")

from service.liveness.domain.face import BoundingBox, FaceCrop, FaceQuality, Point
from service.liveness.domain.report import LivenessReport
from service.liveness.domain.verdict import Verdict


def synthetic_face(
    *,
    area_scale: float = 1.0,
    nose_offset: float = 0.0,
    blur: float = 1.0,
    acceptable: bool | None = None,
) -> FaceCrop:
    """Build a FaceCrop with controllable geometry.

    `area_scale` scales the bbox (proximity); `nose_offset` shifts the
    nose by a fraction of inter-ocular distance (yaw proxy); `blur` sets
    the blur score; `acceptable` overrides the is_acceptable flag.
    """
    w = int(200 * (area_scale ** 0.5))
    h = w
    cx, cy = 200, 200
    bb = BoundingBox(x=cx - w // 2, y=cy - h // 2, width=w, height=h)
    eye_dist = 80 * (area_scale ** 0.5)
    nose_x = cx + nose_offset * eye_dist
    landmarks = (
        Point(x=cx - eye_dist / 2, y=cy - 20),  # right_eye
        Point(x=cx + eye_dist / 2, y=cy - 20),  # left_eye
        Point(x=nose_x, y=cy + 10),             # nose
        Point(x=cx, y=cy + 30),                 # mouth
        Point(x=cx - eye_dist * 1.2, y=cy),     # right_ear
        Point(x=cx + eye_dist * 1.2, y=cy),     # left_ear
    )
    is_ok = (blur >= 0.30) if acceptable is None else acceptable
    quality = FaceQuality(
        blur_score=blur,
        brightness_score=1.0,
        occlusion_score=1.0,
        is_acceptable=is_ok,
    )
    return FaceCrop(
        bbox=bb, landmarks=landmarks, quality=quality,
        crop_path=None, completeness_issues=(),
    )


def passive_sample(verdict: Verdict, score: float = 0.0, reasons=()) -> LivenessReport:
    """Build a minimal LivenessReport for passive-aggregation tests."""
    return LivenessReport(
        request_id="test",
        timestamp=datetime.now(timezone.utc),
        module_version="test",
        thresholds_version="test",
        verdict=verdict,
        score=score,
        reasons=tuple(reasons),
        face_crop=None,
        detectors={},
    )


def feed(session, n, t_start, dt, *, in_oval, occluded=False,
         n_faces=1, eye_blink=None, mouth_smile=None, embedding=None,
         **face_kwargs):
    """Drive `session.submit()` for `n` frames, returning the last snapshot."""
    snap = None
    for i in range(n):
        snap = session.submit(
            synthetic_face(**face_kwargs),
            t_start + i * dt,
            in_oval=in_oval,
            occluded=occluded,
            n_faces_detected=n_faces,
            eye_blink=eye_blink,
            mouth_smile=mouth_smile,
            embedding=embedding,
        )
    return snap


def unit_embedding(seed: int, dim: int = 512):
    """Deterministic L2-normalized embedding for identity tests.

    Different seeds yield near-orthogonal vectors (cosine ~0) modelling
    "different people"; the same seed models the same person.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)
