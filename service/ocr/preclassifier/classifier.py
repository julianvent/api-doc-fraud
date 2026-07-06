import numpy as np

from service.ocr.models import TextLine

from . import aspect, face, mrz_scan
from .models import PreClassResult


_PROOF_OF_ADDRESS_MIN_LINES = 20
_LARGE_FACE_AREA_RATIO      = 0.02


def classify(image: np.ndarray, lines: list[TextLine]) -> PreClassResult:
    image_height             = image.shape[0] if image is not None and image.ndim >= 2 else 0
    mrz_result, mrz_type     = mrz_scan.scan(lines, image_height)
    has_mrz                  = mrz_type is not None
    country_iso              = mrz_result.country if mrz_result and mrz_result.country else None
    aspect_class             = aspect.classify(image)

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

    line_count      = len(lines) if lines else 0
    face_area_ratio = face.largest_face_area_ratio(image)
    face_present    = face_area_ratio > 0.0
    large_face      = face_area_ratio >= _LARGE_FACE_AREA_RATIO

    if line_count >= _PROOF_OF_ADDRESS_MIN_LINES and not large_face:
        return PreClassResult(
            doc_family   = "proof_of_address",
            aspect_class = aspect_class,
            has_face     = face_present,
            confidence   = 0.75 if not face_present else 0.65,
            signals      = {"line_count": line_count, "face_area_ratio": round(face_area_ratio, 4)},
        )

    if large_face:
        return PreClassResult(
            doc_family   = "identity_photo",
            has_face     = True,
            aspect_class = aspect_class,
            confidence   = 0.80,
            signals      = {"face_area_ratio": round(face_area_ratio, 4), "line_count": line_count},
        )

    if face_present:
        return PreClassResult(
            doc_family   = "identity_photo",
            has_face     = True,
            aspect_class = aspect_class,
            confidence   = 0.60,
            signals      = {"face_area_ratio": round(face_area_ratio, 4), "line_count": line_count},
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
        signals    = {"line_count": line_count, "face_area_ratio": round(face_area_ratio, 4)},
    )
