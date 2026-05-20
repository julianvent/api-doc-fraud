

from service.ocr.models import TextLine


LATIN_THRESHOLD      = 0.85
DEVANAGARI_THRESHOLD = 0.85


def _classify_script(text: str) -> str:
    clean = [c for c in text if c.strip()]
    if not clean:
        return "english"

    latin_chars    = sum(1 for c in clean if ord(c) < 0x0900)
    fraction_latin = latin_chars / len(clean)

    if fraction_latin >= LATIN_THRESHOLD:
        return "english"
    elif fraction_latin <= (1 - DEVANAGARI_THRESHOLD):
        return "other"
    return "mixed"


def _extract_latin(text: str) -> str:
    return ''.join(c for c in text if c.strip() and ord(c) < 0x0900).strip()


def filter_latin(lines: list[TextLine]) -> list[TextLine]:
    result = []

    for line in lines:
        script = _classify_script(line.text)

        if script == "english":
            result.append(TextLine(
                text=line.text,
                confidence=line.confidence,
                bbox=line.bbox,
                lang="english"
            ))

        elif script == "mixed":
            latin_text = _extract_latin(line.text)
            if latin_text:
                result.append(TextLine(
                    text=latin_text,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    lang="english",
                    original_text=line.text,
                    was_mixed=True
                ))

    return result