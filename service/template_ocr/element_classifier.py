"""Heuristic classifier that assigns a 'role' hint to each DotsOCR element.

Roles:
  "label"   — the element is likely a field name / title (e.g. "Nombre:", "Apellidos")
  "value"   — the element is likely a field value    (e.g. "Juan Antonio", "1985-05-20")
  "unknown" — not enough signal to decide

Rules are applied in order; the first match wins.
"""

from __future__ import annotations

import re

_DIGIT_ALPHA_RE = re.compile(r"[0-9]")
_ONLY_ALPHA_RE  = re.compile(r"^[^\d]+$")


def _classify_one(element: dict) -> str:
    text = (element.get("text") or "").strip()
    if not text:
        return "unknown"

    # Rule 1 — ends with colon: canonical label pattern ("Nombre:", "Apellidos:")
    if text.endswith(":"):
        return "label"

    words  = text.split()
    n_words = len(words)

    # Rule 2 — short, no digits, short enough to be a field name
    if (
        n_words <= 4
        and len(text) <= 35
        and _ONLY_ALPHA_RE.match(text.replace(" ", "").replace("/", "").replace("-", ""))
        and (text[0].isupper() or text.isupper())
    ):
        return "label"

    # Rule 3 — many words or mix of letters and digits → value
    if n_words > 5 or (bool(_DIGIT_ALPHA_RE.search(text)) and bool(re.search(r"[A-Za-z]", text))):
        return "value"

    # Rule 4 — left-column position and short text → label
    bbox = element.get("bbox") or {}
    x1 = float(bbox.get("x1", 1.0))
    if x1 < 0.35 and n_words <= 3:
        return "label"

    return "unknown"


def classify_elements(elements: list[dict]) -> list[dict]:
    """Return a new list with a 'role' key added to each element dict."""
    return [{**e, "role": _classify_one(e)} for e in elements]
