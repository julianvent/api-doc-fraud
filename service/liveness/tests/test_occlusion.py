"""Tests for the parser-based occlusion detector (pure logic).

We mock the FaceParseResult (segmentation mask) so these run without the
selfie_multiclass model, validating the per-class fraction math and flag
thresholds, not the model itself.
"""

from __future__ import annotations

import numpy as np

from service.liveness.active.face_parser import FaceParseClass, FaceParseResult
from service.liveness.active.occlusion import FaceOcclusionDetector


def _square_face_landmarks(cx=128, cy=128, half=80):
    """478-row landmark array, FACE_OVAL indices forming a square.

    Only the oval indices need real positions; the rest stay at the
    centre (unused by the detector).
    """
    from service.liveness.active.occlusion import _FACE_OVAL_INDICES

    pts = np.full((478, 2), [cx, cy], dtype=np.float32)
    # Lay the oval points around a square boundary.
    n = len(_FACE_OVAL_INDICES)
    for i, idx in enumerate(_FACE_OVAL_INDICES):
        ang = 2 * np.pi * i / n
        pts[idx] = [cx + half * np.cos(ang), cy + half * np.sin(ang)]
    return pts


def _mask_of(value: int, shape=(256, 256)) -> FaceParseResult:
    """A category mask entirely of one class."""
    return FaceParseResult(category_mask=np.full(shape, value, dtype=np.uint8))


def _detect(mask_value):
    det = FaceOcclusionDetector()
    lm = _square_face_landmarks()
    pr = _mask_of(mask_value)
    return det.detect(lm, pr, (256, 256))


def test_all_face_skin_is_clear():
    rep = _detect(FaceParseClass.FACE_SKIN.value)
    assert rep.has_occlusion is False
    assert rep.non_face_skin_fraction < 0.05


def test_all_others_flags_accessory():
    rep = _detect(FaceParseClass.OTHERS.value)
    assert rep.accessory_on_face is True
    assert rep.has_occlusion is True
    assert "glasses" in rep.hint().lower() or "accessory" in rep.hint().lower()


def test_all_clothes_flags_mask():
    rep = _detect(FaceParseClass.CLOTHES.value)
    assert rep.clothes_on_face is True
    assert rep.has_occlusion is True


def test_all_hair_flags_hair():
    rep = _detect(FaceParseClass.HAIR.value)
    assert rep.hair_on_face is True
    assert rep.has_occlusion is True


def test_body_skin_flags_hand():
    rep = _detect(FaceParseClass.BODY_SKIN.value)
    assert rep.body_on_face is True
    assert rep.has_occlusion is True


def test_none_landmarks_is_clear():
    det = FaceOcclusionDetector()
    rep = det.detect(None, _mask_of(FaceParseClass.OTHERS.value), (256, 256))
    assert rep.no_landmarks is True
    assert rep.has_occlusion is False


def test_none_mask_is_clear():
    det = FaceOcclusionDetector()
    rep = det.detect(_square_face_landmarks(), FaceParseResult(category_mask=None), (256, 256))
    assert rep.has_occlusion is False


def test_hint_priority_accessory_over_others():
    # When everything is OTHERS, the accessory hint should win.
    rep = _detect(FaceParseClass.OTHERS.value)
    assert rep.hint() != ""
