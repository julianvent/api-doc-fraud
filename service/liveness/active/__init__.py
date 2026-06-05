"""Active liveness — interactive 3D-motion challenges.

Passive (`liveness.analyze()`) catches replay via forensics; active
catches the rest (pre-recorded video, deepfakes that can't synthesize
unseen viewpoints live, masks) by prompting a random 3D motion within a
short window while passive PAD keeps running on the frames.

Public surface:
  - `ChallengeType`, `ChallengeSpec`, `DEFAULT_POOL`  — taxonomy
  - `ActiveSession` — frame-by-frame state machine
  - `decide_active()` — combine challenge results + passive PAD
  - `ActiveSessionReport` — final audit trail
"""

from service.liveness.active.challenges import (
    DEFAULT_POOL,
    ChallengeSpec,
    ChallengeType,
)
from service.liveness.active.decision import decide_active, passive_verdict
from service.liveness.active.messages import (
    DEFAULT_REJECT_MESSAGE,
    primary_message_for,
    user_message_for,
)
from service.liveness.active.face_embedding import (
    IDENTITY_MATCH_THRESHOLD,
    FaceEmbedder,
    cosine_similarity,
    is_same_identity,
)
from service.liveness.active.face_parser import (
    FaceParseClass,
    FaceParseResult,
    MediaPipeFaceParser,
)
from service.liveness.active.occlusion import (
    FaceOcclusionDetector,
    FaceOcclusionReport,
)
from service.liveness.active.session import ActiveSession, SessionPhase, SessionSnapshot
from service.liveness.domain.active_report import (
    ActiveSessionReport,
    ChallengeResult,
)

__all__ = [
    "ActiveSession",
    "ActiveSessionReport",
    "ChallengeResult",
    "ChallengeSpec",
    "ChallengeType",
    "DEFAULT_POOL",
    "DEFAULT_REJECT_MESSAGE",
    "IDENTITY_MATCH_THRESHOLD",
    "FaceEmbedder",
    "FaceOcclusionDetector",
    "FaceOcclusionReport",
    "FaceParseClass",
    "FaceParseResult",
    "MediaPipeFaceParser",
    "SessionPhase",
    "cosine_similarity",
    "is_same_identity",
    "SessionSnapshot",
    "decide_active",
    "passive_verdict",
    "primary_message_for",
    "user_message_for",
]
