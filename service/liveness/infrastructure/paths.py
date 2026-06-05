"""Filesystem paths and module version."""

from __future__ import annotations

from pathlib import Path

MODULE_ROOT: Path = Path(__file__).resolve().parent.parent
MODULE_VERSION: str = "0.6.0-phase4-demo"


def weights_dir() -> Path:
    """Return the directory where model weights live.

    Created lazily, always under the module root so `service/liveness/`
    is self-contained.
    """
    path = MODULE_ROOT / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(request_id: str | None = None) -> Path:
    """Return the artifact directory for the run.

    With `request_id`, returns (and creates) the per-request subfolder;
    otherwise the top-level outputs root.
    """
    base = MODULE_ROOT / "output"
    if request_id is None:
        base.mkdir(parents=True, exist_ok=True)
        return base
    path = base / request_id
    path.mkdir(parents=True, exist_ok=True)
    return path
