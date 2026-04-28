"""Metadata forensics — standalone POC for document fraud detection."""
from .classifier import Confidence, MetadataReport, classify
from .extractor import MetadataExtractor, MetadataSnapshot
from .rules import Flag, evaluate

__all__ = [
    "MetadataExtractor",
    "MetadataSnapshot",
    "Flag",
    "evaluate",
    "MetadataReport",
    "Confidence",
    "classify",
]
