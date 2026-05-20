"""Localizers: spatial priors (face bbox, etc.). Not forensic detectors."""
from .face import FaceLocalizer, build_face_localizer

__all__ = ["FaceLocalizer", "build_face_localizer"]
