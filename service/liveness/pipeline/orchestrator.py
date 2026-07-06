"""End-to-end orchestration of the liveness pipeline.

Loads input -> localizes face -> runs enabled detectors -> aggregates ->
decides -> writes artifacts -> returns LivenessReport.

Resilience contract: the localizer is mandatory — its warmup/weight
failure propagates (integrator surfaces a 5xx). Detectors are tolerant:
one that fails to warm up is marked DEGRADED and the rest still run;
`decide()` never ACCEPTs while a detector is DEGRADED, forcing REVIEW
with `Reason.DETECTOR_UNAVAILABLE`. Inference-time crashes are handled
inside `Detector.analyze()`, which returns `status=FAILED`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from service.liveness.config.settings import DEFAULT_SETTINGS, Settings
from service.liveness.config.thresholds import CURRENT_THRESHOLDS, ThresholdsConfig
from service.liveness.detectors.minifas import MiniFASDetector
from service.liveness.detectors.moire import MoireDetector
from service.liveness.detectors.protocol import Detector
from service.liveness.domain.face import FaceCrop
from service.liveness.domain.report import DetectorResult, LivenessReport
from service.liveness.domain.verdict import Reason, Status, Verdict
from service.liveness.exceptions import (
    LocalizerError,
    ModelLoadError,
    WeightsHashMismatch,
    WeightsNotFound,
)
from service.liveness.infrastructure.artifacts import (
    ArtifactWriter,
    LocalFilesystemArtifactWriter,
)
from service.liveness.infrastructure.image_io import ImageInput, load_image
from service.liveness.infrastructure.logging import get_logger, request_logger
from service.liveness.infrastructure.paths import MODULE_VERSION, output_dir
from service.liveness.localizers.mediapipe_face import MediaPipeFaceLocalizer
from service.liveness.localizers.protocol import FaceLocalizer
from service.liveness.pipeline.decision import decide
from service.liveness.pipeline.scoring import aggregate_score


_log = get_logger(__name__)


_DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "moire": MoireDetector,
    "minifas": MiniFASDetector,
}


class Orchestrator:
    """Wires the components into a runnable pipeline.

    Composed once at warmup, reused per request — localizer and detectors
    are heavy to construct (model loading), so caching them here matters.
    """

    def __init__(
        self,
        *,
        settings: Settings = DEFAULT_SETTINGS,
        thresholds: ThresholdsConfig = CURRENT_THRESHOLDS,
        localizer: FaceLocalizer | None = None,
        detectors: dict[str, Detector] | None = None,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        self._settings = settings
        self._thresholds = thresholds
        self._localizer = localizer or MediaPipeFaceLocalizer(
            padding=settings.face_crop_padding,
            min_face_size_px=settings.face_min_size_px,
        )
        if detectors is None:
            detectors = {
                name: _DETECTOR_REGISTRY[name]()
                for name in settings.enabled_detectors
                if name in _DETECTOR_REGISTRY
            }
        self._detectors = detectors
        self._degraded: set[str] = set()
        self._artifact_writer = artifact_writer or LocalFilesystemArtifactWriter(output_dir())

    def warmup(self) -> None:
        """Eager-load every model.

        Localizer failure aborts warmup; each detector is tried
        independently, a failure marking it DEGRADED without stopping
        the others.
        """
        try:
            self._localizer.warmup()
        except (WeightsNotFound, WeightsHashMismatch, ModelLoadError) as exc:
            _log.error(
                "warmup.localizer_failed",
                extra={"exc_type": type(exc).__name__, "error": str(exc)},
            )
            raise

        self._degraded.clear()
        for name, detector in self._detectors.items():
            try:
                detector.warmup()
            except (
                WeightsNotFound,
                WeightsHashMismatch,
                ModelLoadError,
            ) as exc:
                self._degraded.add(name)
                _log.error(
                    "warmup.detector_degraded",
                    extra={
                        "detector": name,
                        "exc_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- never let a single detector kill warmup
                self._degraded.add(name)
                _log.exception(
                    "warmup.detector_unexpected_error",
                    extra={"detector": name, "exc_type": type(exc).__name__},
                )
        _log.info(
            "warmup.complete",
            extra={
                "n_detectors": len(self._detectors),
                "degraded": sorted(self._degraded),
            },
        )

    def run(self, source: ImageInput) -> LivenessReport:
        request_id = str(uuid.uuid4())
        log = request_logger(_log, request_id)
        timing: dict[str, float] = {}

        if isinstance(source, (str, Path)):
            source_kind = "path"
        elif isinstance(source, (bytes, bytearray, memoryview)):
            source_kind = "bytes"
        else:
            source_kind = "ndarray"
        log.info("pipeline.start", extra={"source_kind": source_kind})

        t0 = time.perf_counter()
        image = load_image(source)
        timing["load_ms"] = (time.perf_counter() - t0) * 1000.0
        log.debug(
            "image.loaded",
            extra={
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "load_ms": round(timing["load_ms"], 2),
            },
        )

        t0 = time.perf_counter()
        try:
            face = self._localizer.locate(image)
        except LocalizerError:
            # Crashed mid-detection — treat as "no face"; the localizer
            # already logged the exception with a stack trace.
            face = None
        timing["localize_ms"] = (time.perf_counter() - t0) * 1000.0
        if face is not None:
            log.info(
                "face.located",
                extra={
                    "bbox_w": face.bbox.width,
                    "bbox_h": face.bbox.height,
                    "blur": round(face.quality.blur_score, 3),
                    "brightness": round(face.quality.brightness_score, 3),
                    "acceptable": face.quality.is_acceptable,
                    "completeness_issues": list(face.completeness_issues),
                    "localize_ms": round(timing["localize_ms"], 2),
                },
            )

        detector_results: dict[str, DetectorResult] = {}
        face_for_report: FaceCrop | None = None
        evidence_dir: str | None = None
        image_path_str = str(source) if isinstance(source, (str, Path)) else "<bytes>"

        if face is not None:
            crop_pixels = image[
                face.bbox.y : face.bbox.y_max,
                face.bbox.x : face.bbox.x_max,
            ].copy()

            if self._settings.persist_artifacts:
                crop_path = self._artifact_writer.write_crop(request_id, crop_pixels)
                evidence_dir = str(Path(crop_path).parent)
                face_for_report = replace(face, crop_path=crop_path)
            else:
                face_for_report = face

            # Completeness gate BEFORE inference — partial frames / bad
            # geometry are HARD_REJECT-bound, so skip detectors. Logged
            # separately so production sees which check tripped.
            if face_for_report.completeness_issues:
                log.info(
                    "face.incomplete_skipped_detectors",
                    extra={"issues": list(face_for_report.completeness_issues)},
                )
            # Quality gate BEFORE inference — skip detectors on an
            # unusable face. Saves ~30 ms (MiniFAS forward) + ~10 ms
            # (Moire FFT) per request on bad inputs.
            elif face_for_report.quality.is_acceptable:
                for name, detector in self._detectors.items():
                    if name in self._degraded:
                        detector_results[name] = _degraded_result(name)
                        log.warning(
                            "detector.skipped_degraded",
                            extra={"detector": name},
                        )
                        continue
                    t0 = time.perf_counter()
                    result = detector.analyze(
                        crop_pixels, face_for_report, image, image_path_str
                    )
                    timing[f"detect_{name}_ms"] = (time.perf_counter() - t0) * 1000.0
                    detector_results[name] = result
                    log.info(
                        "detector.done",
                        extra={
                            "detector": name,
                            "status": result.status.value,
                            "score": round(result.score, 4),
                            "latency_ms": round(result.latency_ms, 2),
                        },
                    )
            else:
                log.info(
                    "face.quality_failed",
                    extra={
                        "blur": round(face_for_report.quality.blur_score, 3),
                        "brightness": round(
                            face_for_report.quality.brightness_score, 3
                        ),
                    },
                )

        t0 = time.perf_counter()
        score = aggregate_score(detector_results.values())
        verdict, reasons = decide(detector_results, face_for_report, self._thresholds)
        timing["decide_ms"] = (time.perf_counter() - t0) * 1000.0
        log.info(
            "verdict.decided",
            extra={
                "verdict": verdict.value,
                "score": round(score, 4),
                "reasons": list(reasons),
                "degraded": sorted(self._degraded),
            },
        )

        report = LivenessReport(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            module_version=MODULE_VERSION,
            thresholds_version=self._thresholds.version,
            verdict=verdict,
            score=score,
            reasons=reasons,
            face_crop=face_for_report,
            detectors=detector_results,
            timing_ms=timing,
            evidence_dir=evidence_dir,
        )

        if self._settings.persist_artifacts:
            # Ensure the evidence dir exists even with no face, so
            # annotated.jpg + report.html land somewhere the caller can
            # show "we tried but failed".
            if evidence_dir is None:
                annotated_path = self._artifact_writer.write_annotated(request_id, image)
                evidence_dir = str(Path(annotated_path).parent)
                report = replace(report, evidence_dir=evidence_dir)
            else:
                from service.liveness.visualization.overlays import annotate

                annotated = annotate(image, report)
                self._artifact_writer.write_annotated(request_id, annotated)

            self._artifact_writer.write_report(request_id, report)
            self._artifact_writer.write_html(request_id, report)

        return report


def _degraded_result(name: str) -> DetectorResult:
    """Synthesize a DetectorResult for a detector that could not warm up."""
    return DetectorResult(
        name=name,
        status=Status.DEGRADED,
        score=0.0,
        confidence=0.0,
        contribution=Verdict.REVIEW,
        reasons=(Reason.DETECTOR_UNAVAILABLE.value,),
        raw_metrics={},
        heatmap_path=None,
        latency_ms=0.0,
        error="detector failed to warm up; see logs",
    )
