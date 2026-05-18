"""Evaluation over examples/Real and examples/Fake."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from service.tampering.detector import (
    analyze,
    build_engine,
    build_face_localizer,
    build_trufor_engine,
)
from service.tampering.detector.loader import load
from service.tampering.document_crop import locate

EXAMPLES_DIR = _HERE.parent / "examples"
OUTPUT_DIR = _HERE / "service" / "tampering" / "test_output_eval"
ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}


def main() -> None:
    real_dir = EXAMPLES_DIR / "Real"
    fake_dir = EXAMPLES_DIR / "Fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {real_dir} and {fake_dir} to exist"
        )

    real_files = sorted(_collect(real_dir))
    fake_files = sorted(_collect(fake_dir))
    if not real_files or not fake_files:
        raise FileNotFoundError("Real/ or Fake/ has no documents")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading models (one-time)...")
    engine = build_engine(model_name="auto", device="cpu")
    trufor_engine = build_trufor_engine(device="cpu")
    try:
        face_localizer = build_face_localizer()
    except Exception as exc:
        print(f"  [WARN] Face localizer disabled: {exc}")
        face_localizer = None

    total = len(real_files) + len(fake_files)
    print(f"[2/4] Processing {total} document(s) "
          f"({len(real_files)} real / {len(fake_files)} fake)...\n")

    rows = []
    for idx, (path, gt) in enumerate(
        [(p, "Real") for p in real_files] + [(p, "Fake") for p in fake_files], 1
    ):
        print(f"--- [{idx}/{total}] {gt}: {path.name} ---")
        row = _evaluate(
            path, gt, engine, trufor_engine, face_localizer,
            output_dir,
        )
        if row is not None:
            rows.append(row)
            face_tf_str = (
                f"FaceTF={row['face_tf']:.3f} area={row['face_tf_area']:.2%}"
                if row["face_tf_ran"]
                else f"FaceTF=skipped({row['face_tf_skip']})"
            )
            print(
                f"  fraud_score={row['fraud_score']:.3f}  "
                f"label={row['risk_label']:18s}  "
                f"reliability={row['reliability']:6s}  "
                f"findings={row['findings_count']}  "
                f"t={row['total_ms']}ms"
            )
            print(
                f"    DT={row['dt']:.3f}  TF={row['tf']:.3f}  "
                f"area={row['tf_area']:.2%}  {face_tf_str}"
            )
        print()

    print("[3/4] Summary table:\n")
    print(_render_per_doc_table(rows))

    print("\n[4/4] Metrics & calibration:\n")
    _print_confusion(rows)
    _print_distributions(rows)
    _print_suggested_thresholds(rows)
    print(f"\nArtifacts under: {output_dir}")


def _collect(directory: Path):
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]


def _evaluate(
    path, gt, engine, trufor_engine, face_localizer, output_dir,
):
    try:
        pages = load(str(path))
    except Exception as exc:
        print(f"  [ERROR] load failed: {exc}")
        return None

    page_locations = {}
    for page in pages:
        try:
            loc = locate(page.image)
        except Exception as exc:
            print(f"  [WARN] locate failed page {page.page_number}: {exc}")
            loc = None
        if loc is not None:
            page_locations[page.page_number] = loc

    try:
        reports = analyze(
            file_path=str(path),
            engine=engine,
            output_dir=str(output_dir),
            face_localizer=face_localizer,
            enable_face_localizer=face_localizer is not None,
            trufor_engine=trufor_engine,
            enable_trufor=trufor_engine is not None,
            page_locations=page_locations,
        )
    except Exception as exc:
        print(f"  [ERROR] analyze failed: {exc}")
        return None

    # Only consider page 1 — multi-page logic stays the same as before.
    if not reports:
        return None
    r = reports[0]
    return {
        "file": path.name,
        "gt": gt,
        "risk_label": r.risk_label.value,
        "fraud_score": r.fraud_score,
        "reliability": r.reliability.value,
        "findings_count": len(r.findings),
        "total_ms": r.timings.total_ms,
        "dt": r.doctamper.score_mean if r.doctamper.ran else 0.0,
        "dt_outside": r.doctamper.score_outside_face if r.doctamper.ran else 0.0,
        "tf": r.trufor.score if r.trufor.ran else 0.0,
        "tf_area": r.trufor.largest_region_area_fraction if r.trufor.ran else 0.0,
        "face_tf_ran": r.face_trufor.ran,
        "face_tf": r.face_trufor.score if r.face_trufor.ran else 0.0,
        "face_tf_area": (
            r.face_trufor.largest_region_area_fraction if r.face_trufor.ran else 0.0
        ),
        "face_tf_skip": "" if r.face_trufor.ran else (r.face_trufor.skip_reason or ""),
    }


def _render_per_doc_table(rows) -> str:
    headers = (
        "gt", "label", "fraud", "reliab", "DT", "TF", "TF_area",
        "FaceTF", "FaceTF_area", "ms", "file",
    )

    def cell(r, h):
        return {
            "gt": r["gt"],
            "label": r["risk_label"],
            "fraud": f"{r['fraud_score']:.3f}",
            "reliab": r["reliability"],
            "DT": f"{r['dt']:.3f}",
            "TF": f"{r['tf']:.3f}",
            "TF_area": f"{r['tf_area']:.2%}",
            "FaceTF": (f"{r['face_tf']:.3f}" if r["face_tf_ran"] else "skip"),
            "FaceTF_area": (f"{r['face_tf_area']:.2%}" if r["face_tf_ran"] else "skip"),
            "ms": str(r["total_ms"]),
            "file": r["file"],
        }[h]

    widths = [
        max(len(h), max((len(cell(r, h)) for r in rows), default=0))
        for h in headers
    ]
    fmt = "  ".join(f"{{:{w}s}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for r in rows:
        lines.append(fmt.format(*(cell(r, h) for h in headers)))
    return "\n".join(lines)


def _print_confusion(rows) -> None:
    labels = ("LEGITIMATE", "SUSPICIOUS", "LIKELY_MANIPULATED")
    matrix = {(gt, lbl): 0 for gt in ("Real", "Fake") for lbl in labels}
    for r in rows:
        matrix[(r["gt"], r["risk_label"])] += 1

    n_real = sum(matrix[("Real", lbl)] for lbl in labels)
    n_fake = sum(matrix[("Fake", lbl)] for lbl in labels)

    print(f"Risk-label distribution — Real n={n_real}, Fake n={n_fake}")
    print(
        f"  {'class':6s}  {'LEGITIMATE':>10s}  {'SUSPICIOUS':>10s}  "
        f"{'LIKELY_MANIPULATED':>18s}"
    )
    for gt in ("Real", "Fake"):
        legit = matrix[(gt, "LEGITIMATE")]
        susp = matrix[(gt, "SUSPICIOUS")]
        manip = matrix[(gt, "LIKELY_MANIPULATED")]
        print(f"  {gt:6s}  {legit:>10d}  {susp:>10d}  {manip:>18d}")

    if n_real:
        legit_rate = matrix[("Real", "LEGITIMATE")] / n_real
        manip_rate = matrix[("Real", "LIKELY_MANIPULATED")] / n_real
        print(f"\n  Real LEGITIMATE rate:           {legit_rate:.1%}")
        print(f"  Real LIKELY_MANIPULATED rate:   {manip_rate:.1%}  (false positives)")
    if n_fake:
        manip_rate = matrix[("Fake", "LIKELY_MANIPULATED")] / n_fake
        legit_rate = matrix[("Fake", "LEGITIMATE")] / n_fake
        print(f"  Fake LIKELY_MANIPULATED rate:   {manip_rate:.1%}  (true positives)")
        print(f"  Fake LEGITIMATE rate:           {legit_rate:.1%}  (false negatives)")


def _print_distributions(rows) -> None:
    print("\nScore distributions per perito per class:")
    print(f"  {'class':6s} {'perito':10s} {'mean':>7s} {'std':>7s} "
          f"{'p10':>7s} {'p50':>7s} {'p90':>7s} {'min':>7s} {'max':>7s}")
    for gt in ("Real", "Fake"):
        group = [r for r in rows if r["gt"] == gt]
        face_group = [r for r in group if r["face_tf_ran"]]
        for label, key, source in (
            ("fraud_score", "fraud_score", group),
            ("DT", "dt", group),
            ("DT_out", "dt_outside", group),
            ("TF", "tf", group),
            ("TF_area", "tf_area", group),
            ("FaceTF", "face_tf", face_group),
            ("FaceTF_area", "face_tf_area", face_group),
        ):
            values = sorted(r[key] for r in source)
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            p10 = _percentile(values, 10)
            p50 = _percentile(values, 50)
            p90 = _percentile(values, 90)
            print(
                f"  {gt:6s} {label:10s} {mean:>7.3f} {std:>7.3f} "
                f"{p10:>7.3f} {p50:>7.3f} {p90:>7.3f} "
                f"{values[0]:>7.3f} {values[-1]:>7.3f}"
            )


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _print_suggested_thresholds(rows) -> None:
    """Recommend thresholds from empirical distributions.

    Heuristic: choose values that separate Real and Fake distributions:
      * reject_score:   ~ p95 of Real (auto-reject any score above the worst Real)
      * review_score:   ~ p80 of Real (or p10 of Fake) — band start for human review
      * area threshold: ~ p90 of Real TF_area (filter the security-texture noise)
    """
    real = [r for r in rows if r["gt"] == "Real"]
    fake = [r for r in rows if r["gt"] == "Fake"]
    if not real or not fake:
        return

    print("\nSuggested thresholds (from empirical distributions):")
    print("  (auto-reject ~ p95 of Real; review-band start ~ p80 of Real)")

    for label, key, current_reject, current_review in [
        ("DocTamper.score_mean (text_global_reject)", "dt", 0.55, 0.40),
        ("DocTamper.score_outside_face (text_review)", "dt_outside", None, 0.50),
        ("TruFor.score (trufor_reject)", "tf", 0.88, 0.62),
        ("TruFor.area (reject_min_area)", "tf_area", 0.06, 0.02),
        ("FaceTruFor.score (experimental)", "face_tf", None, None),
        ("FaceTruFor.area (experimental)", "face_tf_area", None, None),
    ]:
        if key.startswith("face_tf"):
            real_subset = [r for r in real if r["face_tf_ran"]]
            fake_subset = [r for r in fake if r["face_tf_ran"]]
        else:
            real_subset, fake_subset = real, fake
        real_sorted = sorted(r[key] for r in real_subset)
        fake_sorted = sorted(r[key] for r in fake_subset)
        if not real_sorted or not fake_sorted:
            print(f"  {label}: insufficient data")
            continue
        p95_real = _percentile(real_sorted, 95)
        p80_real = _percentile(real_sorted, 80)
        p10_fake = _percentile(fake_sorted, 10)
        p50_fake = _percentile(fake_sorted, 50)
        current_line = ""
        if current_reject is not None:
            current_line += f"  current reject={current_reject:.2f}"
        if current_review is not None:
            current_line += f"  current review={current_review:.2f}"
        print(
            f"  {label}:\n"
            f"    Real p80={p80_real:.3f}  p95={p95_real:.3f}  |  "
            f"Fake p10={p10_fake:.3f}  p50={p50_fake:.3f}"
            f"{current_line}"
        )


if __name__ == "__main__":
    main()
