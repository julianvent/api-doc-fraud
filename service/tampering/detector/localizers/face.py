"""Face localization via  YuNet detector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..report import DocumentType, FaceDetection

_HERE = Path(__file__).resolve().parent
_WEIGHTS_DIR = _HERE.parent.parent / "weights"
_DEFAULT_ONNX = _WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"

# YuNet runs at its training resolution sweet spot around 320-640 on the long
# side. Documents come in at arbitrary sizes; we downscale to this max side for
# detection and unscale the bbox back to the original resolution.
_DETECTION_MAX_SIDE = 640

# Faces with detector confidence below this are ignored. YuNet is well
# calibrated and 0.6 rejects most false positives on busy document layouts.
_SCORE_THRESHOLD = 0.6
_NMS_THRESHOLD = 0.3
_TOP_K = 50

# Which document types are expected to carry a face. Used only to set
# `FaceDetection.expected`; the verdict rule decides what to do with it.
_FACE_EXPECTED_BY_TYPE = {
    DocumentType.PAN: True,
    DocumentType.INE: True,
    DocumentType.PASSPORT: True,
    DocumentType.VISA: True,
    DocumentType.UNKNOWN: True,
}


class FaceLocalizer:
    """Thin wrapper around `cv2.FaceDetectorYN`."""

    name = "yunet-2023mar"

    def __init__(self, onnx_path: Path = _DEFAULT_ONNX) -> None:
        if not Path(onnx_path).is_file():
            raise RuntimeError(
                f"YuNet ONNX weights not found at {onnx_path}. "
                "Download face_detection_yunet_2023mar.onnx from "
                "https://github.com/opencv/opencv_zoo/tree/main/models/"
                "face_detection_yunet and place it under weights/."
            )
        self._onnx_path = str(onnx_path)
        # Input size is re-set per image via setInputSize; the constructor
        # value is only used before the first detect() call.
        self._detector = cv2.FaceDetectorYN.create(
            model=self._onnx_path,
            config="",
            input_size=(_DETECTION_MAX_SIDE, _DETECTION_MAX_SIDE),
            score_threshold=_SCORE_THRESHOLD,
            nms_threshold=_NMS_THRESHOLD,
            top_k=_TOP_K,
        )

    def locate(
        self, image_rgb: np.ndarray, document_type: DocumentType = DocumentType.UNKNOWN,
    ) -> FaceDetection:
        """Return the highest-confidence face on the page, if any."""
        expected = _FACE_EXPECTED_BY_TYPE.get(document_type, True)
        bbox, confidence = self._detect_largest(image_rgb)
        if bbox is None:
            return FaceDetection(detected=False, expected=expected)
        return FaceDetection(
            detected=True,
            bbox=bbox,
            confidence=float(confidence),
            expected=expected,
        )

    def _detect_largest(
        self, image_rgb: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        h, w = image_rgb.shape[:2]
        scale = min(1.0, _DETECTION_MAX_SIDE / max(h, w))
        if scale < 1.0:
            dw, dh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(image_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            dw, dh = w, h
            resized = image_rgb

        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((dw, dh))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None, 0.0

        # YuNet rows: [x, y, w, h, lm0_x, lm0_y, ..., lm4_x, lm4_y, confidence].
        # Rank by detector confidence (last column) — highest wins.
        best = max(faces, key=lambda row: row[-1])
        fx, fy, fw, fh = best[:4]
        inv = 1.0 / scale if scale > 0 else 1.0
        bbox = (
            max(0, int(round(fx * inv))),
            max(0, int(round(fy * inv))),
            max(1, int(round(fw * inv))),
            max(1, int(round(fh * inv))),
        )
        return bbox, float(best[-1])


_SINGLETON: Optional[FaceLocalizer] = None


def build_face_localizer(onnx_path: Path = _DEFAULT_ONNX) -> FaceLocalizer:
    """Return a process-wide FaceLocalizer, initialized on first call."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = FaceLocalizer(onnx_path=onnx_path)
    return _SINGLETON
