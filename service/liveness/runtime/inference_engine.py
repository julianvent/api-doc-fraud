"""InferenceEngine — runs the vision models concurrently for serving.

Concurrency model: a `ThreadPoolExecutor` of `pool_size` workers, each
holding its OWN models in thread-local storage. This resolves two
hazards of async-server inference: MediaPipe Tasks is NOT thread-safe on
a shared instance (per-thread instances avoid it); and MediaPipe (C++) +
onnxruntime release the GIL during native inference, so threads give
real parallelism without multiprocessing's IPC cost to ship frames.
Sessions run in parallel across the pool; frames of one session must
still be submitted in order (the state machine is stateful).

Public surface: `warmup()` (build each thread's model set),
`process_active(frame, want_embedding, want_occlusion)` -> `FrameSignals`,
`analyze_passive(frame)` -> passive `LivenessReport`, `shutdown()`. The
`process_*_sync` variants run on the calling thread, for tests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from service.liveness.active.face_embedding import FaceEmbedder
from service.liveness.active.face_parser import MediaPipeFaceParser
from service.liveness.active.landmarker import MediaPipeFaceLandmarker
from service.liveness.active.occlusion import FaceOcclusionDetector
from service.liveness.config.settings import DEFAULT_SETTINGS
from service.liveness.domain.report import LivenessReport
from service.liveness.infrastructure.logging import get_logger
from service.liveness.localizers.mediapipe_face import MediaPipeFaceLocalizer
from service.liveness.pipeline.orchestrator import Orchestrator
from service.liveness.runtime.config import DEFAULT_RUNTIME_CONFIG, RuntimeConfig
from service.liveness.runtime.frame_signals import FrameSignals, compute_in_oval


_log = get_logger(__name__)


class _ModelSet:
    """One thread's private set of models. Built once per worker thread."""

    def __init__(self) -> None:
        self.localizer = MediaPipeFaceLocalizer()
        self.landmarker = MediaPipeFaceLandmarker()
        self.parser = MediaPipeFaceParser()
        self.embedder = FaceEmbedder()
        self.occlusion = FaceOcclusionDetector()  # stateless, cheap
        # Passive PAD orchestrator — shares this thread's localizer
        # (sequential, same thread) and never writes artifacts.
        passive_settings = dataclasses.replace(
            DEFAULT_SETTINGS, persist_artifacts=False
        )
        self.passive = Orchestrator(settings=passive_settings, localizer=self.localizer)

    def warmup(self) -> None:
        self.localizer.warmup()
        self.landmarker.warmup()
        self.parser.warmup()
        self.embedder.warmup()
        self.passive.warmup()


class InferenceEngine:
    def __init__(self, config: RuntimeConfig = DEFAULT_RUNTIME_CONFIG) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(
            max_workers=config.pool_size, thread_name_prefix="liveness-infer"
        )
        self._local = threading.local()

    # ── model-set lifecycle ──────────────────────────────────────

    def _models(self) -> _ModelSet:
        ms = getattr(self._local, "models", None)
        if ms is None:
            ms = _ModelSet()
            ms.warmup()
            self._local.models = ms
        return ms

    def warmup(self) -> None:
        """Build every pool thread's model set up-front.

        A barrier forces all `pool_size` workers live at once so each
        builds its own set — otherwise one fast worker could serve every
        warmup task while the others stay cold.
        """
        n = self._config.pool_size
        barrier = threading.Barrier(n)

        def _init() -> None:
            self._models()
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError:
                pass

        futures = [self._executor.submit(_init) for _ in range(n)]
        for f in futures:
            f.result()
        _log.info("inference_engine.ready", extra={"pool_size": n})

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    # ── synchronous cores (run on the current thread) ────────────

    def process_active_sync(
        self,
        frame_bgr: np.ndarray,
        *,
        want_embedding: bool = False,
        want_occlusion: bool = False,
    ) -> FrameSignals:
        ms = self._models()
        h, w = frame_bgr.shape[:2]
        face = ms.localizer.locate(frame_bgr)
        in_oval = compute_in_oval(face, w, h, self._config.oval)

        eye_blink = mouth_smile = None
        occlusion = None
        embedding = None
        if face is not None:
            lm = ms.landmarker.extract(frame_bgr)
            eye_blink = lm.blendshapes.blink_intensity
            mouth_smile = lm.blendshapes.smile_intensity
            if want_occlusion:
                parse = ms.parser.parse(frame_bgr)
                occlusion = ms.occlusion.detect(lm.landmarks_xy, parse, frame_bgr.shape)
            if want_embedding:
                embedding = ms.embedder.embed(frame_bgr, lm.landmarks_xy)

        return FrameSignals(
            face=face,
            in_oval=in_oval,
            eye_blink=eye_blink,
            mouth_smile=mouth_smile,
            occlusion=occlusion,
            embedding=embedding,
        )

    def analyze_passive_sync(self, frame_bgr: np.ndarray) -> LivenessReport:
        return self._models().passive.run(frame_bgr)

    # ── async wrappers (dispatch to the pool) ────────────────────

    async def process_active(
        self,
        frame_bgr: np.ndarray,
        *,
        want_embedding: bool = False,
        want_occlusion: bool = False,
    ) -> FrameSignals:
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self.process_active_sync,
            frame_bgr,
            want_embedding=want_embedding,
            want_occlusion=want_occlusion,
        )
        return await loop.run_in_executor(self._executor, fn)

    async def analyze_passive(self, frame_bgr: np.ndarray) -> LivenessReport:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.analyze_passive_sync, frame_bgr)
