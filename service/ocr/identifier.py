import hashlib
import re

from .backends import VisionBackend


_PROMPT = """Look at this identity document image.
Identify what type of document this is.
Return ONLY a single snake_case word from this list:
visa, passport, national_id, pan_card, aadhaar, driver_license, unknown
Return only the word, nothing else."""


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
    cleaned = raw.strip().lower().split()
    if not cleaned:
        result = "unknown"
    else:
        first = re.sub(r"[^a-z_]", "", cleaned[0])
        result = first or "unknown"

    _cache[key] = result
    return result
