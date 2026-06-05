"""Domain layer — pure data contracts for the liveness module.

Zero framework dependencies (no PyTorch/OpenCV/FastAPI). Every type is a
frozen dataclass or enum, so reports are immutable and JSON-serializable.
"""

from service.liveness.domain.evidence import EngineOutput, HeatmapData
from service.liveness.domain.face import BoundingBox, FaceCrop, FaceQuality, Point
from service.liveness.domain.report import DetectorResult, LivenessReport
from service.liveness.domain.verdict import Reason, Status, Verdict

__all__ = [
    "BoundingBox",
    "DetectorResult",
    "EngineOutput",
    "FaceCrop",
    "FaceQuality",
    "HeatmapData",
    "LivenessReport",
    "Point",
    "Reason",
    "Status",
    "Verdict",
]
