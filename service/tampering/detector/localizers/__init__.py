"""Localizers: find *where* something is in the image (face, signature, MRZ).

Localizers are NOT forensic detectors — they provide spatial priors that
forensic detectors consume. Keep that separation strict: localizer output
must never feed the verdict rule on its own.
"""
from .face import FaceLocalizer, build_face_localizer

__all__ = ["FaceLocalizer", "build_face_localizer"]
