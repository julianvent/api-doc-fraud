"""Infrastructure — every side effect lives here.

Filesystem paths, weight loading, artifact persistence, image I/O. The
rest of the module depends on these via Protocols, staying testable
without real disk.
"""

from service.liveness.infrastructure.artifacts import (
    ArtifactWriter,
    LocalFilesystemArtifactWriter,
)
from service.liveness.infrastructure.image_io import ImageInput, load_image
from service.liveness.infrastructure.loader import resolve as resolve_weight
from service.liveness.infrastructure.manifest import MANIFEST, WeightEntry
from service.liveness.infrastructure.paths import (
    MODULE_ROOT,
    MODULE_VERSION,
    output_dir,
    weights_dir,
)

__all__ = [
    "ArtifactWriter",
    "ImageInput",
    "LocalFilesystemArtifactWriter",
    "MANIFEST",
    "MODULE_ROOT",
    "MODULE_VERSION",
    "WeightEntry",
    "load_image",
    "output_dir",
    "resolve_weight",
    "weights_dir",
]
