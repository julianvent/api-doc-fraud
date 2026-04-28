"""Core preprocessor pipeline: load → preprocess → quality → optional rescue."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .config import Config
from .io.loader import load
from .models import Page, ProcessedPage
from .preprocessing.enhance import enhance
from .preprocessing.preprocess import preprocess
from .quality.assess import assess


def process_file(file_path: str | Path, cfg: Config) -> List[ProcessedPage]:
    """Process a single document end-to-end up to OCR-ready ProcessedPage.

    If the first quality pass fails, retry with a CLAHE + unsharp rescue and
    keep whichever version scores higher.
    """
    pages = load(str(file_path), cfg.io.pdf_render_dpi)
    results: List[ProcessedPage] = []

    for page in pages:
        processed = preprocess(page, cfg.preprocessing)
        report = assess(processed, cfg.quality)

        if not report.passed:
            rescued_img = enhance(processed.image, cfg.preprocessing.enhance)
            rescued_page = Page(
                image=rescued_img,
                page_number=processed.page_number,
                dpi=processed.dpi,
                source=processed.source,
            )
            rescued_report = assess(rescued_page, cfg.quality)
            if rescued_report.score > report.score:
                processed = rescued_page
                report = rescued_report
                report.rescued = True

        results.append(ProcessedPage(
            image=processed.image,
            page_number=processed.page_number,
            dpi=processed.dpi,
            source=processed.source,
            quality=report,
        ))
    return results
