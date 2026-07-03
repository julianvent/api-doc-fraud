from typing import Optional

from service.ocr.models import TextLine, MRZResult
from service.ocr.mrz import detect, MRZ_PATTERN


def _normalize(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "")
        .replace("«", "<")
        .replace("‹", "<")
        .replace("＜", "<")
        .upper()
    )


def _lines_in_bottom_band(lines: list[TextLine], image_height: int, band_ratio: float = 0.2) -> list[TextLine]:
    if not lines or image_height <= 0:
        return list(lines)
    threshold = image_height * (1.0 - band_ratio)
    filtered  = []
    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        ys       = [pt[1] for pt in line.bbox]
        y_center = (min(ys) + max(ys)) / 2
        if y_center >= threshold:
            filtered.append(line)
    return filtered


def _infer_mrz_type(lines: list[TextLine]) -> Optional[str]:
    candidates = []
    for line in lines:
        clean = _normalize(line.text)
        if MRZ_PATTERN.match(clean) and "<" in clean:
            candidates.append(clean)

    if not candidates:
        return None

    lens            = [len(c) for c in candidates]
    has_visa_prefix = any(c.startswith("V") for c in candidates if len(c) in (36, 44))

    if lens.count(44) >= 2:
        return "MRV-A" if has_visa_prefix else "TD3"
    if lens.count(36) >= 2:
        return "MRV-B" if has_visa_prefix else "TD2"
    if lens.count(30) >= 3:
        return "TD1"
    return None


def scan(lines: list[TextLine], image_height: int) -> tuple[Optional[MRZResult], Optional[str]]:
    band_lines = _lines_in_bottom_band(lines, image_height)
    if not band_lines:
        band_lines = list(lines)
    mrz_type   = _infer_mrz_type(band_lines)
    if mrz_type is None:
        return None, None
    mrz_result = detect(band_lines)
    return mrz_result, mrz_type
