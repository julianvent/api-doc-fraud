"""Evaluate the metadata module against examples/Real and examples/Fake.

Runs `service.metadata.extract` over every document in the dataset, maps the
`suspicion_score` to a label (LEGITIMATE / SUSPICIOUS / LIKELY_MANIPULATED)
and prints:
    - Per-document table
    - Confusion matrix Real/Fake vs predicted label
    - Score distribution per class (mean/std/percentiles)
    - Flag coverage per class
    - Suggested thresholds derived from the empirical distributions
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from service.metadata.metadata import extract  # noqa: E402

EXAMPLES_DIR = _HERE.parent / "examples"
ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Thresholds aligned with the classifier's _SEVERITY_PENALTY scale.
SUSPICIOUS_THRESHOLD = 0.25
MANIPULATED_THRESHOLD = 0.50

LABELS = ("LEGITIMATE", "SUSPICIOUS", "LIKELY_MANIPULATED")
CLASSES = ("Real", "Fake")


def main() -> None:
    real_dir = EXAMPLES_DIR / "Real"
    fake_dir = EXAMPLES_DIR / "Fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(f"Expected {real_dir} and {fake_dir}")

    real_files = sorted(_collect(real_dir))
    fake_files = sorted(_collect(fake_dir))
    if not real_files or not fake_files:
        raise FileNotFoundError("Real/ or Fake/ has no supported documents")

    pairs = [(p, "Real") for p in real_files] + [(p, "Fake") for p in fake_files]
    rows = [row for row in (_evaluate(p, gt) for p, gt in pairs) if row is not None]

    print("[1/2] Per-document table:\n")
    print(_render_per_doc_table(rows))

    print("\n[2/2] Metrics and calibration:\n")
    _print_confusion(rows)
    _print_flag_coverage(rows)
    _print_distributions(rows)
    _print_suggested_thresholds(rows)


def _collect(directory: Path) -> list[Path]:
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]


def _evaluate(path: Path, gt: str) -> dict | None:
    try:
        report = extract([path])[0]
    except Exception as exc:
        print(f"[ERROR] extract failed for {path.name}: {exc}", file=sys.stderr)
        return None
    return {
        "file": path.name,
        "gt": gt,
        "format": report.format,
        "suspicion_score": report.suspicion_score,
        "confidence": report.confidence,
        "pred_label": _score_to_label(report.suspicion_score),
        "num_flags": len(report.flags),
        "flag_codes": sorted({f.code for f in report.flags}),
        "summary": report.summary,
    }


def _score_to_label(score: float) -> str:
    if score >= MANIPULATED_THRESHOLD:
        return "LIKELY_MANIPULATED"
    if score >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "LEGITIMATE"


def _render_per_doc_table(rows: list[dict]) -> str:
    headers = ("gt", "pred", "score", "conf", "fmt", "flags", "file")

    def cell(r: dict, h: str) -> str:
        return {
            "gt": r["gt"],
            "pred": r["pred_label"],
            "score": f"{r['suspicion_score']:.3f}",
            "conf": r["confidence"],
            "fmt": r["format"],
            "flags": ",".join(r["flag_codes"]) or "-",
            "file": r["file"],
        }[h]

    widths = [
        max(len(h), max((len(cell(r, h)) for r in rows), default=0))
        for h in headers
    ]
    fmt = "  ".join(f"{{:{w}s}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*(cell(r, h) for h in headers)) for r in rows)
    return "\n".join(lines)


def _print_confusion(rows: list[dict]) -> None:
    matrix = {(gt, lbl): 0 for gt in CLASSES for lbl in LABELS}
    for r in rows:
        matrix[(r["gt"], r["pred_label"])] += 1

    n_real = sum(matrix[("Real", lbl)] for lbl in LABELS)
    n_fake = sum(matrix[("Fake", lbl)] for lbl in LABELS)

    print(f"Predicted-label distribution — Real n={n_real}, Fake n={n_fake}")
    print(
        f"  {'class':6s}  {'LEGITIMATE':>10s}  {'SUSPICIOUS':>10s}  "
        f"{'LIKELY_MANIPULATED':>18s}"
    )
    for gt in CLASSES:
        print(
            f"  {gt:6s}  {matrix[(gt, 'LEGITIMATE')]:>10d}  "
            f"{matrix[(gt, 'SUSPICIOUS')]:>10d}  "
            f"{matrix[(gt, 'LIKELY_MANIPULATED')]:>18d}"
        )

    if n_real:
        print(f"\n  Real LEGITIMATE rate:         {matrix[('Real', 'LEGITIMATE')] / n_real:.1%}")
        print(f"  Real SUSPICIOUS rate:         {matrix[('Real', 'SUSPICIOUS')] / n_real:.1%}  (soft false positives)")
        print(f"  Real LIKELY_MANIPULATED rate: {matrix[('Real', 'LIKELY_MANIPULATED')] / n_real:.1%}  (false positives)")
    if n_fake:
        print(f"  Fake LIKELY_MANIPULATED rate: {matrix[('Fake', 'LIKELY_MANIPULATED')] / n_fake:.1%}  (true positives)")
        print(f"  Fake SUSPICIOUS rate:         {matrix[('Fake', 'SUSPICIOUS')] / n_fake:.1%}  (soft alerts)")
        print(f"  Fake LEGITIMATE rate:         {matrix[('Fake', 'LEGITIMATE')] / n_fake:.1%}  (false negatives)")

    # Binarized view: any alert (SUSPICIOUS or LIKELY_MANIPULATED) counts as positive.
    if n_real and n_fake:
        tp = matrix[("Fake", "SUSPICIOUS")] + matrix[("Fake", "LIKELY_MANIPULATED")]
        fn = matrix[("Fake", "LEGITIMATE")]
        fp = matrix[("Real", "SUSPICIOUS")] + matrix[("Real", "LIKELY_MANIPULATED")]
        tn = matrix[("Real", "LEGITIMATE")]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("\n  Binarized (any alert = positive):")
        print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"    precision={precision:.3f}  recall={recall:.3f}  "
              f"F1={f1:.3f}  accuracy={accuracy:.3f}")


def _print_flag_coverage(rows: list[dict]) -> None:
    print("\nFlag coverage per class (docs that fire each rule):")
    for gt in CLASSES:
        group = [r for r in rows if r["gt"] == gt]
        if not group:
            continue
        counter: Counter[str] = Counter()
        for r in group:
            counter.update(r["flag_codes"])
        if not counter:
            print(f"  {gt}: no flags fired")
            continue
        n = len(group)
        print(f"  {gt} (n={n}):")
        for code, count in counter.most_common():
            print(f"    {code:35s}  {count:3d}  ({count / n:.1%})")


def _print_distributions(rows: list[dict]) -> None:
    print("\nsuspicion_score distribution per class:")
    print(f"  {'class':6s} {'n':>3s} {'mean':>7s} {'std':>7s} "
          f"{'p10':>7s} {'p50':>7s} {'p90':>7s} {'min':>7s} {'max':>7s}")
    for gt in CLASSES:
        values = sorted(r["suspicion_score"] for r in rows if r["gt"] == gt)
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(
            f"  {gt:6s} {len(values):>3d} {mean:>7.3f} {std:>7.3f} "
            f"{_percentile(values, 10):>7.3f} {_percentile(values, 50):>7.3f} "
            f"{_percentile(values, 90):>7.3f} "
            f"{values[0]:>7.3f} {values[-1]:>7.3f}"
        )

    print("\nConfidence distribution per class:")
    for gt in CLASSES:
        group = [r for r in rows if r["gt"] == gt]
        if not group:
            continue
        conf_counter = Counter(r["confidence"] for r in group)
        breakdown = "  ".join(
            f"{lvl}={conf_counter.get(lvl, 0)}"
            for lvl in ("high", "medium", "indeterminate")
        )
        print(f"  {gt}: {breakdown}")


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _print_suggested_thresholds(rows: list[dict]) -> None:
    real = sorted(r["suspicion_score"] for r in rows if r["gt"] == "Real")
    fake = sorted(r["suspicion_score"] for r in rows if r["gt"] == "Fake")
    if not real or not fake:
        return
    print("\nSuggested thresholds on suspicion_score:")
    print(f"  Real:  p80={_percentile(real, 80):.3f}  "
          f"p95={_percentile(real, 95):.3f}  max={real[-1]:.3f}")
    print(f"  Fake:  p10={_percentile(fake, 10):.3f}  "
          f"p50={_percentile(fake, 50):.3f}  min={fake[0]:.3f}")
    print(f"  Current: SUSPICIOUS>={SUSPICIOUS_THRESHOLD:.2f}  "
          f"LIKELY_MANIPULATED>={MANIPULATED_THRESHOLD:.2f}")
    print("  Heuristic: review-band ~ p80 Real; auto-reject ~ p95 Real if < p10 Fake")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    main()
