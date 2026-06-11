import numpy as np

from service.ocr.models import TextLine

from . import aspect, face, mrz_scan
from .models import PreClassResult


# Minimum OCR lines to call something "text-heavy" (proof of address).
# Utility bills typically have 30-80 lines; identity docs have 8-15.
_PROOF_OF_ADDRESS_MIN_LINES = 20

# Face area as a fraction of total image area. Holder photos on passports/IDs
# are typically 5-20%; logos with face-like features are usually well under 2%.
_LARGE_FACE_AREA_RATIO = 0.02


def classify(image: np.ndarray, lines: list[TextLine]) -> PreClassResult:
    """Heuristic document family classifier.

    Decision order (Option C):
      1. MRZ detected           → identity_mrz (most reliable signal)
      2. Many text lines AND face is small or absent → proof_of_address
      3. Large face present     → identity_photo (passport / ID / license)
      4. Small face present     → identity_photo (fallback)
      5. Card-shaped aspect     → identity_card
      6. otherwise              → unknown

    Aspect ratio is no longer the primary signal — photos taken with phones
    rarely match A4/Letter exactly and we got too many false positives. The
    two reliable signals are: how much text the OCR found, and whether a
    prominent face is visible.
    """
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

    line_count      = len(lines) if lines else 0
    face_area_ratio = face.largest_face_area_ratio(image)
    face_present    = face_area_ratio > 0.0
    large_face      = face_area_ratio >= _LARGE_FACE_AREA_RATIO

    # Text-heavy doc without a prominent holder photo → proof of address.
    # Small face logos are tolerated.
    if line_count >= _PROOF_OF_ADDRESS_MIN_LINES and not large_face:
        return PreClassResult(
            doc_family   = "proof_of_address",
            aspect_class = aspect_class,
            has_face     = face_present,
            confidence   = 0.75 if not face_present else 0.65,
            signals      = {
                "line_count":      line_count,
                "face_area_ratio": round(face_area_ratio, 4),
            },
        )

    # Prominent face → almost certainly an identity doc with a holder photo.
    if large_face:
        return PreClassResult(
            doc_family   = "identity_photo",
            has_face     = True,
            aspect_class = aspect_class,
            confidence   = 0.80,
            signals      = {
                "face_area_ratio": round(face_area_ratio, 4),
                "line_count":      line_count,
            },
        )

    # Some face but small, and not text-heavy → likely a card-style identity doc.
    if face_present:
        return PreClassResult(
            doc_family   = "identity_photo",
            has_face     = True,
            aspect_class = aspect_class,
            confidence   = 0.60,
            signals      = {
                "face_area_ratio": round(face_area_ratio, 4),
                "line_count":      line_count,
            },
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
