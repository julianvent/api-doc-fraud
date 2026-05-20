"""Constraint sets used by rules to detect contradictions.

These are not preference lists. Each set describes deterministic behavior of
a tool or format, so a violation of that behavior becomes a falsifiable flag.
"""
from __future__ import annotations

# Government issuers known to digitally sign documents. Only positive signal.
TRUSTED_GOVT: frozenset[str] = frozenset({
    "digilocker", "uidai", "cca india", "cca-india",
    "nsdl", "protean", "emudhra",
})

# Vendors that always emit MakerNote in EXIF. Missing MakerNote = contradiction.
ALWAYS_WRITES_MAKERNOTE: frozenset[str] = frozenset({
    "apple", "canon", "nikon", "sony",
})

# Editors that always write xmpMM:History. Absent history = contradiction.
LEAVES_XMP_HISTORY: frozenset[str] = frozenset({
    "adobe photoshop", "photoshop", "lightroom", "adobe lightroom",
    "affinity photo",
})

# Image editors whose mere presence in Software/producer is a signal for
# documents that should originate from a camera or government scanner.
IMAGE_EDITORS: frozenset[str] = frozenset({
    "adobe photoshop", "photoshop", "lightroom", "adobe lightroom",
    "affinity photo", "affinity designer",
    "gimp", "pixelmator", "krita", "paint.net", "paint shop pro",
    "corel", "inkscape", "canva", "figma", "sketch",
    "acdsee", "luminar", "topaz",
})

# PNG/JPEG info keys whose presence is itself the signal of generative origin.
AI_GENERATION_CHUNK_KEYS: frozenset[str] = frozenset({
    "parameters",
    "prompt",
    "negative_prompt",
    "workflow",
    "sd-metadata",
    "invokeai_metadata",
    "comfy_workflow",
    "novelai",
    "Dream",
})

# C2PA action codes that explicitly declare generative AI authorship.
C2PA_AI_ACTION_MARKERS: tuple[bytes, ...] = (
    b"c2pa.ai_generative",
    b"c2pa.synthesizing",
    b"c2pa.training_mining",
    b"c2pa.ai_training",
    b"gen-ai-assertions",
    b"genAiInference",
    b"ai_generative",
)

# IPTC digitalSourceType values that imply non-camera, synthetic origin.
# Reference: http://cv.iptc.org/newscodes/digitalsourcetype/
C2PA_SYNTHETIC_SOURCE_TYPES: tuple[bytes, ...] = (
    b"trainedAlgorithmicMedia",
    b"compositeWithTrainedAlgorithmicMedia",
    b"algorithmicMedia",
    b"compositeCapture",
    b"composite",
)

# Bytes that, if present anywhere in the file, indicate a C2PA manifest.
C2PA_PRESENCE_MARKERS: tuple[bytes, ...] = (
    b"urn:c2pa:",
    b"xmlns:c2pa",
    b"JUMBF",
    b"caBX",
    b"c2pa.actions",
    b"c2pa.assertions",
)


def contains_any(haystack: str | None, needles: frozenset[str]) -> str | None:
    """Return the first needle found in haystack (case-insensitive), else None."""
    if not haystack:
        return None
    lowered = haystack.lower()
    for n in needles:
        if n in lowered:
            return n
    return None
