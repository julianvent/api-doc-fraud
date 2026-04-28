"""Resolve module-relative paths against the preprocessor root, not the CWD."""
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = MODULE_ROOT / "config"
