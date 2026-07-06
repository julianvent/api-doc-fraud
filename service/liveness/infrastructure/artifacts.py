"""Artifact persistence — Protocol-based so storage backend can swap.

Phase 2 ships only `LocalFilesystemArtifactWriter`. A future S3-backed
implementation plugs into the same Protocol.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from service.liveness.domain.report import LivenessReport


class ArtifactWriter(Protocol):
    """Persists per-request evidence."""

    def write_crop(self, request_id: str, image: np.ndarray) -> str:
        ...

    def write_heatmap(self, request_id: str, name: str, heatmap: np.ndarray) -> str:
        ...

    def write_report(self, request_id: str, report: LivenessReport) -> str:
        ...

    def write_annotated(self, request_id: str, image: np.ndarray) -> str:
        ...

    def write_html(self, request_id: str, report: LivenessReport) -> str:
        ...


class LocalFilesystemArtifactWriter:
    """Writes to `<root>/<request_id>/`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _dir(self, request_id: str) -> Path:
        path = self._root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_crop(self, request_id: str, image: np.ndarray) -> str:
        path = self._dir(request_id) / "crop.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write crop to {path}")
        return str(path)

    def write_heatmap(self, request_id: str, name: str, heatmap: np.ndarray) -> str:
        path = self._dir(request_id) / f"heatmap_{name}.png"
        if heatmap.dtype != np.uint8:
            heatmap = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
        if not cv2.imwrite(str(path), heatmap):
            raise RuntimeError(f"Failed to write heatmap to {path}")
        return str(path)

    def write_report(self, request_id: str, report: LivenessReport) -> str:
        path = self._dir(request_id) / "report.json"
        payload = dataclasses.asdict(report)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        return str(path)

    def write_annotated(self, request_id: str, image: np.ndarray) -> str:
        path = self._dir(request_id) / "annotated.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write annotated image to {path}")
        return str(path)

    def write_html(self, request_id: str, report: LivenessReport) -> str:
        # Imported lazily so the infrastructure layer does not depend
        # on visualization at import time.
        from service.liveness.visualization.html_report import write as write_html_report

        return str(write_html_report(report, self._dir(request_id)))


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):  # Enum
        return value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")
