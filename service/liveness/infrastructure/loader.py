"""Weight loading + hash verification.

The loader is the single point that validates manifest entries against
disk. Engines and detectors must call `resolve()` to get a verified
path — never read manifest paths directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from service.liveness.exceptions import WeightsHashMismatch, WeightsNotFound
from service.liveness.infrastructure.logging import get_logger
from service.liveness.infrastructure.manifest import WeightEntry, get


_log = get_logger(__name__)


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve(name: str) -> Path:
    """Return the verified absolute path for a manifest entry.

    Raises:
        WeightsNotFound: file is missing — run
            `python -m service.liveness.scripts.download_weights <name>`.
        WeightsHashMismatch: file is present but its SHA256 differs from manifest.
    """
    entry: WeightEntry = get(name)
    path = entry.absolute_path()
    if not path.exists():
        _log.error(
            "weight.missing",
            extra={"weight": name, "expected_path": str(path)},
        )
        raise WeightsNotFound(
            f"Weight {name!r} missing at {path}. "
            f"Run: python -m service.liveness.scripts.download_weights {name}"
        )
    if entry.sha256 is not None:
        actual = _sha256_of(path)
        if actual != entry.sha256:
            _log.error(
                "weight.hash_mismatch",
                extra={
                    "weight": name,
                    "path": str(path),
                    "expected_sha256": entry.sha256,
                    "actual_sha256": actual,
                },
            )
            raise WeightsHashMismatch(
                f"Weight {name!r} at {path} sha256={actual} "
                f"does not match manifest sha256={entry.sha256}"
            )
    _log.debug("weight.resolved", extra={"weight": name, "path": str(path)})
    return path
