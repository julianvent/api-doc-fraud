"""Resolve module-relative paths against the OCR root, not the CWD."""
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = MODULE_ROOT / "config"
ASSETS_DIR = MODULE_ROOT / "assets"
