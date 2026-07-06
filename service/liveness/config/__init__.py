"""Configuration — runtime settings and calibrated thresholds.

`settings` is environment-dependent (detectors to run, paths).
`thresholds` is a calibration artifact tied to a dataset version. They
evolve on different timescales, hence separate files.
"""

from service.liveness.config.settings import Settings, DEFAULT_SETTINGS
from service.liveness.config.thresholds import (
    CURRENT_THRESHOLDS,
    DetectorThresholds,
    EnsembleThresholds,
    FaceQualityThresholds,
    ThresholdsConfig,
)

__all__ = [
    "CURRENT_THRESHOLDS",
    "DEFAULT_SETTINGS",
    "DetectorThresholds",
    "EnsembleThresholds",
    "FaceQualityThresholds",
    "Settings",
    "ThresholdsConfig",
]
