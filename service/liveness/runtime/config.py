"""Centralized runtime/server configuration for liveness.

One source of truth for the serving knobs: inference pool size,
transmission rate/resolution hints, oval geometry (client draws, server
checks), and timeouts. Scale concurrency by changing `pool_size` /
`max_concurrent_sessions` here; nothing else moves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OvalGeometry:
    """Face-guide oval as fractions of the frame (resolution-independent)."""

    cx: float = 0.5
    cy: float = 0.5
    rx: float = 0.18
    ry: float = 0.30
    # Face-width vs oval-width band that counts as "good distance".
    face_width_ratio_min: float = 0.65
    face_width_ratio_max: float = 1.15


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    # ── concurrency ──────────────────────────────────────────────
    # Inference worker threads, each with its own model set. Raise to
    # scale; keep `pool_size <= physical cores`.
    pool_size: int = 2
    # Active sessions served at once. The registry already handles N, so
    # this is the only number to change.
    max_concurrent_sessions: int = 1

    # ── transmission hints (sent to the client) ─────────────────
    send_fps: int = 15
    send_width: int = 1280
    send_height: int = 720

    # ── timing ───────────────────────────────────────────────────
    # Max parser cadence per session; other frames reuse the last
    # occlusion result.
    parser_throttle_ms: int = 300
    # Whole-session wall-clock cap.
    session_timeout_s: float = 60.0
    # No frame for this long mid-session -> state machine sees face_lost.
    frame_idle_timeout_s: float = 1.5

    # ── geometry ─────────────────────────────────────────────────
    oval: OvalGeometry = OvalGeometry()

    # ── persistence ──────────────────────────────────────────────
    # Server sessions do NOT write per-frame artifacts to disk.
    persist_artifacts: bool = False


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()
