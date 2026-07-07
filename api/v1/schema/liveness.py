"""Shared liveness verdict vocabulary.
Used by the WebSocket result message (api/v1/schema/liveness_ws.py).
"""
from __future__ import annotations

from enum import Enum


class LivenessVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
