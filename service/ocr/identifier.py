import hashlib
import re

from .backends import VisionBackend


_PROMPT = """Look at this identity document image.
Identify what type of document this is.

Return a short snake_case identifier in English (e.g. passport, visa,
driver_license, permanent_resident_card, residence_permit, voter_id,
diplomatic_passport, military_id). Use the most specific common term.

If you cannot determine the document type confidently, return: unknown.

Return ONLY the snake_case identifier, nothing else."""


_cache: dict[str, str] = {}


def _image_hash(image_path: str) -> str:
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def identify(image_path: str, backend: VisionBackend) -> str:
    key = _image_hash(image_path)
    if key in _cache:
        return _cache[key]

    raw = backend.describe(image_path, _PROMPT)
    first_line = (raw.strip().splitlines() or [""])[0].lower()
    normalized = re.sub(r"[^a-z_]+", "_", first_line)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    result = normalized or "unknown"

    _cache[key] = result
    return result
