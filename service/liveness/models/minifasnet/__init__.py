"""Vendored MiniFASNet architecture.

Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
License: Apache 2.0 (per upstream repo).

Only MiniFASNetV2 is exposed (the variant for our shipped
2.7_80x80_MiniFASNetV2.pth checkpoint). Other upstream variants were
trimmed; restore them from the source repo if a new checkpoint needs them.
"""

from service.liveness.models.minifasnet.minifasnet import MiniFASNetV2, get_kernel

__all__ = ["MiniFASNetV2", "get_kernel"]
