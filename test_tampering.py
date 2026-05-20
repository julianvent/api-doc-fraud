from __future__ import annotations

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

DEFAULT_TARGET = _HERE.parent / "examples" / "Real" / "passan.pdf"
OUTPUT_DIR = _HERE / "service" / "tampering" / "test_output"

ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET

    if target.is_file():
        if target.suffix.lower() not in ALLOWED_EXTS:
            raise ValueError(
                f"Unsupported extension {target.suffix!r} — "
                f"allowed: {sorted(ALLOWED_EXTS)}"
            )
        files = [target]
    elif target.is_dir():
        files = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No documents in {target}")
    else:
        raise FileNotFoundError(f"Target not found: {target}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading models (one-time)...")
    engine = build_engine(model_name="auto", device="cpu")
    trufor_engine = build_trufor_engine(device="cpu")
    try:
        face_localizer = build_face_localizer()
    except Exception as exc:
        print(f"  [WARN] Face localizer disabled: {exc}")
        face_localizer = None

    print(f"[2/3] Processing {len(files)} document(s)...\n")
    rows = []
    for idx, image_path in enumerate(files, 1):
        print(f"--- [{idx}/{len(files)}] {image_path.name} ---")
        try:
            pages = load(str(image_path))
        except Exception as exc:
            print(f"  [ERROR] load failed: {exc}")
            rows.append((image_path.name, "-", "LOAD_ERROR", "-", "-", "-"))
            continue

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
                file_path=str(image_path),
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
            rows.append((image_path.name, "-", "ANALYZE_ERROR", "-", "-", "-"))
            continue

        for report in reports:
            page_id = f"p{report.metadata.page_index}/{report.metadata.page_count}"
            dt = f"{report.doctamper.score_mean:.3f}" if report.doctamper.ran else "skip"
            tf = f"{report.trufor.score:.3f}" if report.trufor.ran else "skip"
            face_tf = (
                f"{report.face_trufor.score:.3f}"
                if report.face_trufor.ran else "skip"
            )
            label = report.risk_label.value
            fraud = f"{report.fraud_score:.3f}"
            reliab = report.reliability.value
            print(
                f"  {page_id}: {label:18s} fraud={fraud}  "
                f"reliab={reliab}  DT={dt}  TF={tf}  FaceTF={face_tf}"
            )
            rows.append((
                image_path.name, page_id, label, fraud, reliab, dt, tf, face_tf,
            ))
        print()

    print(f"[3/3] Summary ({len(rows)} page(s)):\n")
    print(_render_table(rows))
    print(f"\nArtifacts under: {output_dir}")


def _render_table(rows) -> str:
    headers = ("file", "page", "label", "fraud", "reliab", "DT", "TF", "FaceTF")
    widths = [
        max(len(headers[i]), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:{w}s}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for r in rows:
        lines.append(fmt.format(*(str(c) for c in r[:len(headers)])))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
