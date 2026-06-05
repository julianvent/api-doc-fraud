"""Face-occlusion detection via semantic segmentation.

Replaces the prior heuristic stack (Laplacian variance, HSV saturation,
cheek-anchored skin fraction) with ONE direct signal: per-pixel parsing
from MediaPipe selfie_multiclass. Rule: inside the face-oval polygon
(from FaceLandmarker), the fraction of non-FACE_SKIN pixels is the
occlusion measure; the per-class breakdown says which object to remove.

The 6 classes emitted by the segmenter are:

    0 BACKGROUND
    1 HAIR
    2 BODY_SKIN   (neck, shoulders, hand)
    3 FACE_SKIN   <- acceptable inside the face oval
    4 CLOTHES     (masks, scarves, hoodies, hand wearing a glove)
    5 OTHERS      (glasses, hats, microphones, headphones, accessories)

Per-class thresholds are tuned against the internal bona-fide dataset
(n=41). Adjust if the deployment camera / population pushes baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from service.liveness.active.face_parser import FaceParseClass, FaceParseResult
from service.liveness.infrastructure.logging import get_logger


_log = get_logger(__name__)


# FaceLandmarker FACEMESH_FACE_OVAL indices — outer face boundary.
_FACE_OVAL_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)


# Per-class minimum fraction (of total face-oval pixels) that flips
# the corresponding flag. Tuned conservatively so single stray pixels
# at the oval boundary don't fire spurious hints.
_ACCESSORY_MIN_FRACTION = 0.020   # OTHERS: glasses ~2-5%, hats partial
_CLOTHES_MIN_FRACTION = 0.060     # CLOTHES: masks > 30%; threshold low
_HAIR_MIN_FRACTION = 0.080        # HAIR on face: forehead hair / fringe
_BODY_SKIN_MIN_FRACTION = 0.080   # BODY_SKIN inside face = hand on face

# Aggregate threshold for the generic "object on face" flag. A clean
# face has ~3-8% non-face-skin pixels (model is imperfect at the oval
# boundary, mislabels eyebrows/lips); > 12% means a real object covers
# the face.
_NON_FACE_SKIN_MIN_FRACTION = 0.12


@dataclass(frozen=True, slots=True)
class FaceOcclusionReport:
    """Direct face-occlusion result derived from face parsing.

    Per-class flags let the demo show a specific hint ("Remove your hat")
    rather than a generic one. `has_occlusion` is True if any class flag
    is set OR the aggregate non-face-skin fraction crosses the threshold.
    """

    object_on_face: bool
    hair_on_face: bool
    clothes_on_face: bool
    accessory_on_face: bool
    body_on_face: bool
    non_face_skin_fraction: float
    class_fractions: dict[str, float] = field(default_factory=dict)
    no_landmarks: bool = False

    @property
    def has_occlusion(self) -> bool:
        return self.object_on_face

    def hint(self) -> str:
        """User-facing instruction. Most specific class wins."""
        if self.accessory_on_face:
            return "Remove glasses, hat or any accessory from your face"
        if self.clothes_on_face:
            return "Remove the mask or cloth covering your face"
        if self.hair_on_face:
            return "Move your hair away from your face"
        if self.body_on_face:
            return "Move your hand away from your face"
        if self.object_on_face:
            return "Remove any object covering your face"
        return ""


_CLEAR_RESULT = FaceOcclusionReport(
    object_on_face=False,
    hair_on_face=False,
    clothes_on_face=False,
    accessory_on_face=False,
    body_on_face=False,
    non_face_skin_fraction=0.0,
    class_fractions={},
    no_landmarks=True,
)


class FaceOcclusionDetector:
    """Stateless detector — call `detect()` per frame.

    Inputs: `landmarks_xy` (FaceLandmarker (N,2), builds the face-oval
    polygon mask), `parse_result` (MediaPipeFaceParser.parse() output,
    sampled inside the polygon), `image_shape` ((H,W), aligns the
    input-resolution category mask with the landmark polygon).
    """

    name = "face_occlusion"

    def detect(
        self,
        landmarks_xy: np.ndarray | None,
        parse_result: FaceParseResult,
        image_shape: tuple[int, ...],
    ) -> FaceOcclusionReport:
        if landmarks_xy is None or parse_result.category_mask is None:
            return _CLEAR_RESULT
        if len(landmarks_xy) <= max(_FACE_OVAL_INDICES):
            return _CLEAR_RESULT

        # Build face-oval mask at the original image resolution.
        oval_mask = _build_face_oval_mask(image_shape, landmarks_xy)
        oval_pixels = int((oval_mask > 0).sum())
        if oval_pixels < 200:
            return _CLEAR_RESULT

        # Align the category mask to image resolution. MediaPipe
        # returns the mask already at input resolution, but with a
        # trailing channel dimension that we squeeze.
        category_mask = parse_result.category_mask
        if category_mask.ndim == 3:
            category_mask = category_mask.squeeze(-1)
        H, W = image_shape[:2]
        if category_mask.shape != (H, W):
            category_mask = cv2.resize(
                category_mask, (W, H), interpolation=cv2.INTER_NEAREST
            )

        # Per-class pixel counts INSIDE the face oval.
        in_oval = oval_mask > 0
        counts = {
            c: int(((category_mask == int(c.value)) & in_oval).sum())
            for c in FaceParseClass
        }
        fractions = {c.name: counts[c] / oval_pixels for c in FaceParseClass}

        non_face_skin = oval_pixels - counts[FaceParseClass.FACE_SKIN]
        non_face_skin_fraction = non_face_skin / oval_pixels

        # Per-class flags (specific hints).
        accessory = fractions[FaceParseClass.OTHERS.name] > _ACCESSORY_MIN_FRACTION
        clothes = fractions[FaceParseClass.CLOTHES.name] > _CLOTHES_MIN_FRACTION
        hair = fractions[FaceParseClass.HAIR.name] > _HAIR_MIN_FRACTION
        body = fractions[FaceParseClass.BODY_SKIN.name] > _BODY_SKIN_MIN_FRACTION

        # Generic flag: any specific OR aggregate threshold crossed.
        object_on_face = (
            accessory
            or clothes
            or hair
            or body
            or non_face_skin_fraction > _NON_FACE_SKIN_MIN_FRACTION
        )

        return FaceOcclusionReport(
            object_on_face=object_on_face,
            hair_on_face=hair,
            clothes_on_face=clothes,
            accessory_on_face=accessory,
            body_on_face=body,
            non_face_skin_fraction=non_face_skin_fraction,
            class_fractions=fractions,
        )


def _build_face_oval_mask(
    image_shape: tuple[int, ...], landmarks_xy: np.ndarray
) -> np.ndarray:
    """Binary mask: 255 inside the FACEMESH_FACE_OVAL polygon, 0 outside."""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    poly = landmarks_xy[list(_FACE_OVAL_INDICES)].astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return mask
