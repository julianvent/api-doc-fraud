from typing import Optional

from service.ocr.models import TextLine
from service.ocr.preclassifier import PreClassResult
from service.template_ocr.schema import Fingerprint


_TOP_LINES_FOR_QUERY = 12


def _top_lines(lines: list[TextLine], n: int) -> list[str]:
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda l: -float(l.confidence or 0.0))
    return [l.text.strip() for l in sorted_lines[:n] if l.text and l.text.strip()]


def serialize_for_query(preclass: PreClassResult, lines: list[TextLine]) -> str:
    """
    Builds the text representation of an INCOMING document (verify path),
    using signals from the preclassifier and the OCR layout.
    This is the string that gets embedded and queried against Qdrant.
    """
    parts = [f"family={preclass.doc_family}"]
    if preclass.mrz_type:
        parts.append(f"mrz_type={preclass.mrz_type}")
    if preclass.country_iso:
        parts.append(f"country={preclass.country_iso}")
    if preclass.aspect_class:
        parts.append(f"aspect={preclass.aspect_class}")

    header = "Document signals: " + ", ".join(parts)
    anchors = _top_lines(lines, _TOP_LINES_FOR_QUERY)
    body = "Top OCR lines:\n" + "\n".join(f"- {l}" for l in anchors) if anchors else ""

    return f"{header}\n{body}".strip()


def serialize_for_template(
    fingerprint: Optional[Fingerprint],
    preclass: Optional[PreClassResult] = None,
) -> str:
    """
    Builds the text representation of a TEMPLATE (upload path).
    Uses the VLM-generated fingerprint (layout_desc + anchors) and optional preclass signals.
    """
    parts = []
    if preclass:
        if preclass.doc_family:
            parts.append(f"family={preclass.doc_family}")
        if preclass.mrz_type:
            parts.append(f"mrz_type={preclass.mrz_type}")
        if preclass.country_iso:
            parts.append(f"country={preclass.country_iso}")

    header = "Document signals: " + ", ".join(parts) if parts else ""

    body_parts: list[str] = []
    if fingerprint:
        if fingerprint.layout_desc:
            body_parts.append(f"Layout: {fingerprint.layout_desc}")
        if fingerprint.anchors:
            body_parts.append("Anchors:\n" + "\n".join(f"- {a}" for a in fingerprint.anchors))

    body = "\n".join(body_parts)
    return f"{header}\n{body}".strip() if header else body.strip()
