"""Face localizers — detection + crop + quality gate."""

from service.liveness.localizers.mediapipe_face import MediaPipeFaceLocalizer
from service.liveness.localizers.protocol import FaceLocalizer

__all__ = ["FaceLocalizer", "MediaPipeFaceLocalizer"]
