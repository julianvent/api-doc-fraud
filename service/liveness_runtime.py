"""Process-wide singletons for active liveness: the inference engine
and the session registry.

The `InferenceEngine` (shared model pool) is created and warmed once at
startup. The `SessionRegistry` enforces the concurrency cap — today 1,
but it already tracks N sessions, so scaling is a config change only.

Registry operations are synchronous and free of `await`, so they are
atomic under the single-threaded asyncio event loop — no lock needed.
"""
from __future__ import annotations

from service.liveness.runtime import InferenceEngine
from service.liveness.runtime.config import DEFAULT_RUNTIME_CONFIG, RuntimeConfig


class SessionRegistry:
    """Tracks active session ids and enforces `max_concurrent_sessions`."""

    def __init__(self, max_sessions: int) -> None:
        self._max = max_sessions
        self._active: set[str] = set()

    def try_acquire(self, session_id: str) -> bool:
        """Reserve a slot. Returns False when at capacity (atomic in the loop)."""
        if len(self._active) >= self._max:
            return False
        self._active.add(session_id)
        return True

    def release(self, session_id: str) -> None:
        self._active.discard(session_id)

    @property
    def active_count(self) -> int:
        return len(self._active)


_config: RuntimeConfig = DEFAULT_RUNTIME_CONFIG
_engine: InferenceEngine | None = None
_registry = SessionRegistry(_config.max_concurrent_sessions)


def get_config() -> RuntimeConfig:
    return _config


def get_engine() -> InferenceEngine:
    if _engine is None:
        raise RuntimeError("liveness inference engine not warmed up")
    return _engine


def get_registry() -> SessionRegistry:
    return _registry


def warmup() -> None:
    """Build and warm the shared inference engine. Call once at startup."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine(_config)
        _engine.warmup()


def shutdown() -> None:
    """Drain the inference pool. Call once at shutdown."""
    global _engine
    if _engine is not None:
        _engine.shutdown()
        _engine = None
