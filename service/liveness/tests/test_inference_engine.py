"""Integration tests for the inference engine.

These load the REAL models (slower) and require weights present
(download_weights); a module-scoped engine fixture warms up once. The
key test, test_concurrent_processing_*, validates the concurrency
design: per-thread MediaPipe instances must not crash or corrupt each
other under parallel load.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

import cv2
import pytest

from service.liveness.runtime import InferenceEngine, FrameSignals
from service.liveness.runtime.config import RuntimeConfig
from service.liveness.domain.report import LivenessReport


# Dataset ships next to the package; resolve relative to this test file.
_DATASET = Path(__file__).resolve().parent.parent / "dataset" / "bona_fide"
_FACE_IMG = sorted(str(p) for p in _DATASET.glob("WIN_*.jpg"))


def _load_face():
    if not _FACE_IMG:
        pytest.skip("no dataset face images available")
    img = cv2.imread(_FACE_IMG[0])
    if img is None:
        pytest.skip("could not read dataset face image")
    return img


@pytest.fixture(scope="module")
def engine():
    eng = InferenceEngine(RuntimeConfig(pool_size=2))
    eng.warmup()
    yield eng
    eng.shutdown()


# ── basic signal production ───────────────────────────────────────

def test_process_active_returns_signals(engine):
    sig = engine.process_active_sync(_load_face())
    assert isinstance(sig, FrameSignals)
    assert sig.face is not None              # a face is present
    assert isinstance(sig.in_oval, bool)
    # blendshapes computed whenever a face is found
    assert sig.eye_blink is not None
    assert sig.mouth_smile is not None


def test_embedding_gate(engine):
    img = _load_face()
    assert engine.process_active_sync(img, want_embedding=False).embedding is None
    emb = engine.process_active_sync(img, want_embedding=True).embedding
    assert emb is not None and emb.shape == (512,)


def test_occlusion_gate(engine):
    img = _load_face()
    assert engine.process_active_sync(img, want_occlusion=False).occlusion is None
    occ = engine.process_active_sync(img, want_occlusion=True).occlusion
    assert occ is not None  # a FaceOcclusionReport


def test_analyze_passive_returns_report(engine):
    report = engine.analyze_passive_sync(_load_face())
    assert isinstance(report, LivenessReport)
    assert report.verdict is not None


def test_no_face_image_returns_safe_signals(engine):
    import numpy as np
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    sig = engine.process_active_sync(blank)
    assert sig.face is None
    assert sig.in_oval is False
    assert sig.embedding is None


# ── concurrency (validates the per-thread model design) ───────────

def test_concurrent_sync_no_crash(engine):
    """Many parallel sync calls across the pool must not crash or
    corrupt results (per-thread MediaPipe instances)."""
    img = _load_face()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(engine.process_active_sync, img, want_embedding=True, want_occlusion=True)
            for _ in range(12)
        ]
        results = [f.result() for f in futures]
    assert all(isinstance(r, FrameSignals) and r.face is not None for r in results)
    # Same image -> embeddings should be (near) identical across threads.
    import numpy as np
    embs = [r.embedding for r in results]
    base = embs[0]
    assert all(float(np.dot(base, e)) > 0.99 for e in embs)


def test_async_gather(engine):
    """The async wrappers dispatch to the pool and gather correctly."""
    img = _load_face()

    async def _run():
        return await asyncio.gather(
            *[engine.process_active(img) for _ in range(6)]
        )

    results = asyncio.run(_run())
    assert all(isinstance(r, FrameSignals) and r.face is not None for r in results)
