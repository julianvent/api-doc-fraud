"""Detector adapters — one per spoof signal.

Each detector wraps an engine and translates `EngineOutput` into a
`DetectorResult` against the module's report contract.
"""

from service.liveness.detectors.minifas import MiniFASDetector
from service.liveness.detectors.moire import MoireDetector
from service.liveness.detectors.protocol import Detector

__all__ = ["Detector", "MiniFASDetector", "MoireDetector"]
