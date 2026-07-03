"""WebSocket message schemas for the active-liveness session."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from api.v1.schema.liveness import LivenessVerdict


class OvalSchema(BaseModel):
    cx: float
    cy: float
    rx: float
    ry: float


class SessionStartedMsg(BaseModel):
    type: str = "session_started"
    session_id: str
    challenges: List[str]
    oval: OvalSchema
    send_fps: int
    send_width: int
    send_height: int
    max_session_seconds: float


class ChallengeInfo(BaseModel):
    index: int
    total: int
    type: Optional[str] = None
    prompt: Optional[str] = None


class SnapshotMsg(BaseModel):
    type: str = "snapshot"
    phase: str
    challenge: Optional[ChallengeInfo] = None
    challenge_phase: int = 0          # 0 = reach peak, 1 = return to center
    time_remaining_s: float = 0.0
    in_oval: bool = False
    wrong_direction: bool = False
    occlusion_hint: Optional[str] = None
    preflight_progress: float = 0.0


class ResultMsg(BaseModel):
    type: str = "result"
    session_id: str
    verdict: LivenessVerdict          # ACCEPT | REJECT (binary)
    reasons: List[str]
    message: str


class ErrorMsg(BaseModel):
    type: str = "error"
    code: str                          # busy | bad_frame | timeout | internal
    message: str
