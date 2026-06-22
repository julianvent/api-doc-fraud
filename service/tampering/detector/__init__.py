"""Tampering detection module. Produces a continuous `fraud_score` per page;
the consumer applies its own accept/reject policy.
"""
from .detectors import DocTamperDetector, TruForDetector
from .engine import (
    DocTamperEngine,
    MockEngine,
    TamperingEngine,
    TruForEngine,
    build_engine,
    build_trufor_engine,
)
from .loader import Page, load
from .localizers import FaceLocalizer, build_face_localizer
from .pipeline import analyze
from .report import (
    Artifacts,
    DocTamperResult,
    DocumentLocation,
    DocumentType,
    ExecutionMetadata,
    FaceDetection,
    Finding,
    PageReport,
    Region,
    Reliability,
    RiskLabel,
    Severity,
    Timings,
    TruForResult,
    Zone,
    format_report,
)
from .scoring import score
from .thresholds import DEFAULT, Thresholds
from .visualization import save_face_crop, save_heatmap, save_overlay

__all__ = [
    "Page",
    "load",
    "TamperingEngine",
    "MockEngine",
    "DocTamperEngine",
    "TruForEngine",
    "build_engine",
    "build_trufor_engine",
    "DocTamperDetector",
    "TruForDetector",
    "FaceLocalizer",
    "build_face_localizer",
    "RiskLabel",
    "Reliability",
    "Severity",
    "Finding",
    "Timings",
    "Zone",
    "DocumentType",
    "Region",
    "FaceDetection",
    "DocTamperResult",
    "DocumentLocation",
    "TruForResult",
    "Artifacts",
    "ExecutionMetadata",
    "PageReport",
    "format_report",
    "Thresholds",
    "DEFAULT",
    "analyze",
    "score",
    "save_heatmap",
    "save_overlay",
    "save_face_crop",
]
