from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from service.preprocessor.app.io.loader import SUPPORTED_ALL
from service.preprocessor.preprocessor import process

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
OUTPUT_DIR = Path(__file__).resolve().parent / "validate_scan_output"


def _collect_files(args: list[str]) -> list[Path]:
    if not args:
        if not EXAMPLES_DIR.is_dir():
            return []
        return sorted(
            p for p in EXAMPLES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_ALL
        )

    files: list[Path] = []
    for a in args:
        path = Path(a)
        if path.is_dir():
            files.extend(
                p for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_ALL
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_ALL:
            files.append(path)
        else:
            print(f"  skipping (not a supported file or dir): {path}")
    return files


def main(argv: list[str]) -> int:
    files = _collect_files(argv)
    if not files:
        print("No supported files found.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, bool, bool, float, str, float]] = []
    print(f"\nProcessing {len(files)} files...\n")

    for path in files:
        t0 = time.perf_counter()
        try:
            pages = process([path])
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR {path.name}: {exc}")
            rows.append((path.name, 0, False, False, 0.0, f"error: {exc}", elapsed))
            continue
        elapsed = time.perf_counter() - t0

        for page in pages:
            label = f"{path.stem}_p{page.page_number}.png"
            cv2.imwrite(str(OUTPUT_DIR / label), page.image)
            rows.append((
                path.name,
                page.page_number,
                page.scanned,
                page.quality.passed,
                page.quality.score,
                page.quality.reason,
                elapsed,
            ))

    _print_report(rows)
    print(f"\nOutput images written to: {OUTPUT_DIR}\n")
    return 0


def _print_report(rows: list[tuple[str, int, bool, bool, float, str, float]]) -> None:
    print(f"{'file':<22}{'pg':>3}  {'scan':>4}  {'pass':>4}  {'score':>6}  {'time(s)':>7}  reason")
    print("-" * 100)
    n_pages = len(rows)
    n_scanned = sum(1 for r in rows if r[2])
    n_passed = sum(1 for r in rows if r[3])
    for name, pg, scanned, passed, score, reason, elapsed in rows:
        print(
            f"{name[:22]:<22}{pg:>3}  "
            f"{'Y' if scanned else '-':>4}  "
            f"{'Y' if passed else '-':>4}  "
            f"{score:>6.3f}  "
            f"{elapsed:>7.2f}  "
            f"{reason}"
        )
    print("-" * 100)
    print(
        f"total pages: {n_pages}  |  scanned: {n_scanned} ({n_scanned/n_pages:.0%})  "
        f"|  quality passed: {n_passed} ({n_passed/n_pages:.0%})"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
