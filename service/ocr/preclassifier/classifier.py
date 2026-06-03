import numpy as np

from service.ocr.models import TextLine

from . import aspect, face, mrz_scan
from .models import PreClassResult


_PROOF_OF_ADDRESS_MIN_LINES = 25


def _is_paper_aspect(image: np.ndarray) -> bool:
    """A4 / Letter (vertical or horizontal). Excludes TD1 card aspects."""
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    if h == 0:
        return False
    ratio = w / h
    # A4 vertical (~0.707), Letter vertical (~0.773)
    if 0.67 <= ratio <= 0.80:
        return True
    # A4 horizontal (~1.414), Letter horizontal (~1.294)
    if 1.25 <= ratio <= 1.45:
        return True
    return False


def classify(image: np.ndarray, lines: list[TextLine]) -> PreClassResult:
    image_height = image.shape[0] if image is not None and image.ndim >= 2 else 0

    mrz_result, mrz_type = mrz_scan.scan(lines, image_height)
    has_mrz              = mrz_type is not None
    country_iso          = mrz_result.country if mrz_result and mrz_result.country else None
    aspect_class         = aspect.classify(image)

    if has_mrz:
        return PreClassResult(
            doc_family   = "identity_mrz",
            country_iso  = country_iso,
            mrz_type     = mrz_type,
            has_face     = False,
            aspect_class = aspect_class,
            confidence   = 0.95 if (mrz_result and mrz_result.valid) else 0.85,
            signals      = {"mrz_valid": bool(mrz_result and mrz_result.valid)},
        )

    line_count   = len(lines) if lines else 0
    paper_aspect = _is_paper_aspect(image)
    face_present = face.has_face(image)

    if paper_aspect and line_count >= _PROOF_OF_ADDRESS_MIN_LINES and not face_present:
        return PreClassResult(
            doc_family   = "proof_of_address",
            aspect_class = aspect_class,
            has_face     = False,
            confidence   = 0.70,
            signals      = {"paper_aspect": True, "line_count": line_count},
        )

    if face_present:
        return PreClassResult(
            doc_family   = "identity_photo",
            has_face     = True,
            aspect_class = aspect_class,
            confidence   = 0.75,
            signals      = {"face_detected": True, "line_count": line_count},
        )

    if aspect_class == "TD1":
        return PreClassResult(
            doc_family   = "identity_card",
            aspect_class = aspect_class,
            confidence   = 0.55,
            signals      = {"td1_aspect": True, "line_count": line_count},
        )

    return PreClassResult(
        doc_family = "unknown",
        confidence = 0.0,
        signals    = {"line_count": line_count},
    )
