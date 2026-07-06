"""Facade — the stable public surface of the liveness module."""

from __future__ import annotations

from service.liveness.domain.report import LivenessReport
from service.liveness.infrastructure.image_io import ImageInput
from service.liveness.pipeline.orchestrator import Orchestrator


_ORCHESTRATOR: Orchestrator | None = None


def _instance() -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = Orchestrator()
    return _ORCHESTRATOR


def warmup() -> None:
    """Eagerly construct singletons and load weights.

    Call from the FastAPI lifespan handler on startup so the first
    request avoids model-load latency.
    """
    _instance().warmup()


def analyze(source: ImageInput) -> LivenessReport:
    """Run liveness analysis on a single image.

    Accepts:
      - `str` / `Path` — image file on disk.
      - `bytes` — JPEG/PNG/etc. encoded buffer (e.g. from FastAPI UploadFile).
      - `np.ndarray` — already-decoded BGR uint8 (H, W, 3).
    """
    return _instance().run(source)


def reset() -> None:
    """Drop the cached singletons.

    Intended for tests; production callers should not need this.
    """
    global _ORCHESTRATOR
    _ORCHESTRATOR = None
