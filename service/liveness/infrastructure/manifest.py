"""Weight manifest — authoritative source for what to download.

Each entry pins a file by URL + SHA256. `download_weights.py` fetches it
under `weights/`; the loader verifies the hash before returning the path,
raising `WeightsHashMismatch` rather than use a tampered/partial weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from service.liveness.infrastructure.paths import weights_dir


@dataclass(frozen=True, slots=True)
class WeightEntry:
    """One downloadable artifact.

    When `archive_member` is set, `url` is a .zip and that member is
    extracted to `relative_path`; the pinned `sha256` is the EXTRACTED
    member's hash, not the archive's.
    """

    name: str
    relative_path: str  # under weights_dir(), forward slashes
    url: str
    sha256: str | None  # None = unverified (placeholder until first download confirms)
    license: str
    size_bytes: int | None
    notes: str = ""
    archive_member: str | None = None  # member to extract when url is a zip

    def absolute_path(self) -> Path:
        return weights_dir() / Path(self.relative_path)


MANIFEST: dict[str, WeightEntry] = {
    "blaze_face_short_range": WeightEntry(
        name="blaze_face_short_range",
        relative_path="mediapipe/blaze_face_short_range.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        ),
        sha256="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        license="Apache-2.0",
        size_bytes=229746,
        notes="MediaPipe BlazeFace short-range detector. Used by localizers/mediapipe_face.py.",
    ),
    "minifas_v2_2_7_80x80": WeightEntry(
        name="minifas_v2_2_7_80x80",
        relative_path="minifas/2.7_80x80_MiniFASNetV2.pth",
        url=(
            "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/"
            "raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
        ),
        sha256="a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
        license="Apache-2.0",
        size_bytes=1849453,
        notes=(
            "MiniFASNetV2 anti-spoof checkpoint trained at scale=2.7, "
            "input 80x80. Used by detectors/minifas.py."
        ),
    ),
    "face_landmarker": WeightEntry(
        name="face_landmarker",
        relative_path="mediapipe/face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        license="Apache-2.0",
        size_bytes=3758596,
        notes=(
            "MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes). "
            "Used by active/landmarker.py to detect blink and smile."
        ),
    ),
    "selfie_multiclass": WeightEntry(
        name="selfie_multiclass",
        relative_path="mediapipe/selfie_multiclass_256x256.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        ),
        sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
        license="Apache-2.0",
        size_bytes=16371837,
        notes=(
            "MediaPipe selfie multiclass segmenter. Outputs 6 classes per pixel: "
            "background, hair, body-skin, face-skin, clothes, others (glasses/hat/etc). "
            "Used by active/face_parser.py to detect ANY object covering the face."
        ),
    ),
    "face_recognition": WeightEntry(
        name="face_recognition",
        relative_path="insightface/w600k_mbf.onnx",
        url=(
            "https://github.com/deepinsight/insightface/releases/download/"
            "v0.7/buffalo_s.zip"
        ),
        archive_member="w600k_mbf.onnx",
        sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        license="MIT",
        size_bytes=13616099,
        notes=(
            "MobileFaceNet (ArcFace-trained), insightface buffalo_s. 512-d face "
            "embedding for identity-consistency across an active session. "
            "Used by active/face_embedding.py. Measured on the internal dataset: "
            "same-person cosine median 0.70, different-person max 0.30 (clean split)."
        ),
    ),
}


def get(name: str) -> WeightEntry:
    entry = MANIFEST.get(name)
    if entry is None:
        raise KeyError(f"Unknown weight entry: {name!r}")
    return entry
