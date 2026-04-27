"""Facade for the metadata module. Consumed by the orchestrator."""
from __future__ import annotations

from pathlib import Path

from service.metadata.analyzer import MetadataExtractor, MetadataReport, classify

_extractor = MetadataExtractor()


def extract(paths: list[Path | str]) -> list[MetadataReport]:
    """Run metadata forensics on every file. Returns one report per file."""
    return [classify(_extractor.extract(p)) for p in paths]
