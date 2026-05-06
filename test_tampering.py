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
    build_mvssnet_engine,
    format_report,
)

IMAGE_PATH = _HERE.parent / "examples" / "visaad.jpeg"
OUTPUT_DIR = _HERE / "service" / "tampering" / "test_output"

def main() -> None:
    image_path = Path(IMAGE_PATH)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading models...")
    engine = build_engine(model_name="auto", device="cpu")
    mvssnet_engine = build_mvssnet_engine(device="cpu")
    try:
        face_localizer = build_face_localizer()
    except Exception as exc:
        print(f"  [WARN] Face localizer disabled: {exc}")
        face_localizer = None

    print(f"[2/3] Analyzing: {image_path.name}")
    reports = analyze(
        file_path=str(image_path),
        engine=engine,
        output_dir=str(output_dir),
        face_localizer=face_localizer,
        enable_face_localizer=face_localizer is not None,
        mvssnet_engine=mvssnet_engine,
        enable_mvssnet=mvssnet_engine is not None,
    )

    print(f"[3/3] Done. {len(reports)} page(s) processed.\n")
    for report in reports:
        print(format_report(report))
        print()

    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
