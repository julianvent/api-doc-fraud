"""Runtime layer — web-framework-agnostic primitives for serving liveness.

`InferenceEngine` runs the vision models concurrently (thread pool, one
model set per thread); `FrameSignals` is its per-frame result for an
`ActiveSession`. A web server builds session orchestration on top; these
carry no web dependency themselves.
"""

from service.liveness.runtime.config import RuntimeConfig, DEFAULT_RUNTIME_CONFIG
from service.liveness.runtime.frame_signals import FrameSignals, compute_in_oval
from service.liveness.runtime.inference_engine import InferenceEngine
from service.liveness.runtime.session_driver import (
    ActiveSessionDriver,
    DriverResult,
    DriverUpdate,
)

__all__ = [
    "ActiveSessionDriver",
    "DEFAULT_RUNTIME_CONFIG",
    "DriverResult",
    "DriverUpdate",
    "FrameSignals",
    "InferenceEngine",
    "RuntimeConfig",
    "compute_in_oval",
]
