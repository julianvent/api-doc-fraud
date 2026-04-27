"""Forensic detectors — each produces a result dataclass for `PageReport`.

A detector reads the page (and optional localizer output), runs its model,
and returns a frozen result object plus any heatmap needed for visualization.
Detectors never take verdict decisions — that is `detector.decision`'s job.
"""
from .doctamper import DocTamperDetector
from .mvssnet import MVSSNetDetector

__all__ = ["DocTamperDetector", "MVSSNetDetector"]
