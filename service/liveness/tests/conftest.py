"""Pytest configuration for the liveness suite.

Sets quiet logging env vars before any test imports MediaPipe/TF. The
reusable factories (synthetic_face, passive_sample, feed) live in
`factories.py`, imported directly by the tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LIVENESS_LOG_LEVEL", "WARNING")
