"""Forensic detectors. Each returns a result dataclass + optional heatmap;
scoring lives in `scoring.py`.
"""
from .doctamper import DocTamperDetector
from .trufor import TruForDetector

__all__ = ["DocTamperDetector", "TruForDetector"]
