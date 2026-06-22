"""Face localization data contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """2D landmark coordinate in pixel space."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned face bounding box in pixel space.

    Coordinates follow image convention: x grows right, y grows down.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x_max(self) -> int:
        return self.x + self.width

    @property
    def y_max(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class FaceQuality:
    """Per-face quality metrics used by the quality gate.

    All scores normalized to [0, 1]. Higher is better.
    """

    blur_score: float
    brightness_score: float
    occlusion_score: float
    is_acceptable: bool


@dataclass(frozen=True, slots=True)
class FaceCrop:
    """Output of the face localizer + crop stage.

    `completeness_issues` carries Reason strings for structural problems:
    bbox at image edge, keypoints outside the unpadded bbox, or impossible
    geometry (e.g. mouth above eyes). Non-empty => face unusable, and
    `decide()` issues HARD_REJECT.
    """

    bbox: BoundingBox
    landmarks: tuple[Point, ...]
    quality: FaceQuality
    crop_path: str | None
    completeness_issues: tuple[str, ...] = ()
