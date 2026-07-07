"""Exception hierarchy for the liveness module."""

from __future__ import annotations


class LivenessError(Exception):
    """Base class for all module errors."""


class WeightsNotFound(LivenessError):
    """A required model weight file is missing or unreadable."""


class WeightsHashMismatch(LivenessError):
    """A weight file is present but its SHA256 does not match the manifest."""


class ModelLoadError(LivenessError):
    """A detector or localizer failed during warmup (load + state_dict)."""


class LocalizerError(LivenessError):
    """The localizer raised during face detection."""


class InferenceError(LivenessError):
    """A detector raised during forward pass / postprocessing."""


class EngineInferenceError(InferenceError):
    """Backwards-compatible alias for `InferenceError`."""


class FaceNotDetected(LivenessError):
    """No face was detected in the input image."""


class LowQualityInput(LivenessError):
    """The face was detected but failed the quality gate."""


class InvalidImageInput(LivenessError):
    """Input image violates boundary invariants (size, dtype, channels)."""


class ConfigurationError(LivenessError):
    """Settings or thresholds are inconsistent or invalid."""
