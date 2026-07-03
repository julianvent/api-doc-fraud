"""Image input normalization — accepts path, bytes, or ndarray.

The module works on `np.ndarray` BGR uint8 (OpenCV convention); anything
else routes through `load_image()`. Boundary invariants enforced here:
side 64–8192px, <= 64MP, <= 25MB encoded, aspect 1/10–10/1, 3-channel,
uint8. Pathological inputs are rejected so downstream assumes sane shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from service.liveness.exceptions import InvalidImageInput

ImageInput = Union[str, Path, bytes, np.ndarray]


# Boundary constants — chosen to reject obviously-bad inputs without
# blocking realistic eKYC captures.
MIN_SIDE_PX = 64
MAX_SIDE_PX = 8192
MAX_PIXELS = 64_000_000          # ~64MP — guards against decompression bombs
MAX_BYTES = 25 * 1024 * 1024     # 25 MB encoded
MIN_ASPECT = 0.10                # 1:10 (very tall)
MAX_ASPECT = 10.0                # 10:1 (very wide)


def load_image(source: ImageInput) -> np.ndarray:
    """Return an HxWx3 BGR uint8 image from any supported input.

    Raises LivenessError if the input cannot be decoded or violates the
    boundary invariants documented at module level.
    """
    if isinstance(source, np.ndarray):
        return _validate_array(source)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise InvalidImageInput(f"Image not found: {path}")
        size = path.stat().st_size
        if size == 0:
            raise InvalidImageInput(f"Empty image file: {path}")
        if size > MAX_BYTES:
            raise InvalidImageInput(
                f"Image file too large: {size:,} bytes (max {MAX_BYTES:,})"
            )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageInput(f"Could not decode image at {path}")
        return _validate_array(image)

    if isinstance(source, (bytes, bytearray, memoryview)):
        if len(source) == 0:
            raise InvalidImageInput("Empty image bytes")
        if len(source) > MAX_BYTES:
            raise InvalidImageInput(
                f"Image bytes too large: {len(source):,} bytes (max {MAX_BYTES:,})"
            )
        buf = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageInput("Could not decode image from bytes")
        return _validate_array(image)

    raise InvalidImageInput(f"Unsupported image input type: {type(source).__name__}")


def _validate_array(image: np.ndarray) -> np.ndarray:
    """Enforce shape, dtype and dimension invariants on a decoded image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageInput(f"Expected HxWx3 BGR image, got shape={image.shape}")
    if image.dtype != np.uint8:
        raise InvalidImageInput(f"Expected uint8 image, got dtype={image.dtype}")

    h, w = image.shape[:2]
    if h < MIN_SIDE_PX or w < MIN_SIDE_PX:
        raise InvalidImageInput(
            f"Image too small: {w}x{h} (min side {MIN_SIDE_PX}px)"
        )
    if h > MAX_SIDE_PX or w > MAX_SIDE_PX:
        raise InvalidImageInput(
            f"Image too large: {w}x{h} (max side {MAX_SIDE_PX}px)"
        )
    if h * w > MAX_PIXELS:
        raise InvalidImageInput(
            f"Image has too many pixels: {h * w:,} (max {MAX_PIXELS:,})"
        )
    aspect = w / h
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        raise InvalidImageInput(
            f"Image aspect ratio extreme: {aspect:.2f} "
            f"(allowed {MIN_ASPECT}–{MAX_ASPECT})"
        )
    return image
