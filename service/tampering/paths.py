"""Module-relative paths anchored to the tampering root, not the CWD."""
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = MODULE_ROOT / "weights"
OUTPUT_DIR = MODULE_ROOT / "output"
