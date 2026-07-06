#!/usr/bin/env python3
"""Debug overlay: run the generate pipeline on real images and draw element boxes.

Usage (modo OCR — corre preprocesador + PaddleOCR):
    python debug_overlay.py <image1> [<image2> ...] [--out-dir DIR]

Usage (modo template — dibuja regiones del template guardado sin OCR):
    python debug_overlay.py --template <template.json> <imagen_preprocesada> [...]

Modo OCR (default):
  1. Corre el preprocesador (misma pipeline que generate).
  2. Corre PaddleOCR.
  3. Envuelve la salida como DetectedElements.
  4. Dibuja un polígono gris + ID sobre cada región de texto detectada.
  5. Guarda la imagen anotada en --out-dir (default: overlay_output/).

Modo template (--template PATH):
  1. Lee la imagen directamente (sin preprocesador).
  2. Carga el JSON del template (label_region / value_region por campo).
  3. Dibuja label_region en azul ("L:key") y value_region en verde ("V:key").
  4. Guarda la imagen anotada en --out-dir.

Note on DocAligner (warp of perspective):
  The preprocessor includes a DocAligner scanner that detects the document
  boundary quad and warps it to a fronto-parallel view. This is the same
  transformation that runs at verify time, where each incoming document is
  registered image-to-image against the stored template image. The warp is
  therefore what makes coordinates reusable across instances, not a problem.
  The QA in Step 4 measures alignment consistency margin (how much a box drifts
  between two instances of the same template), not whether the approach is viable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


def process_image(img_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    from service.preprocessor import preprocessor
    from service.ocr.models import Config as OCRConfig
    from service.ocr.ocr import _get_engine
    from service.template_ocr.elements import textlines_to_elements
    from service.template_ocr.overlay import draw_element_boxes

    pages = preprocessor.process([img_path])
    if not pages:
        print(f"  SKIP: preprocessor returned no pages for {img_path.name}", file=sys.stderr)
        return

    np_img = _ensure_rgb(pages[0].image)

    # Save raw preprocessed image for side-by-side comparison.
    preprocessed_path = out_dir / f"{img_path.stem}_preprocessed.png"
    cv2.imwrite(str(preprocessed_path), cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

    engine = _get_engine(OCRConfig())
    lines = engine.extract(np_img)
    elements = textlines_to_elements(lines)

    annotated = draw_element_boxes(np_img, elements)
    overlay_path = out_dir / f"{img_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(
        f"  {img_path.name}: {len(elements)} elements"
        f" → {overlay_path.name}, {preprocessed_path.name}"
    )


def process_template_image(
    img_path: pathlib.Path,
    template_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    from service.template_ocr.schema import Template
    from service.template_ocr.overlay import draw_template_regions

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  SKIP: cannot read {img_path}", file=sys.stderr)
        return
    img_rgb = _ensure_rgb(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    template = Template.model_validate(json.loads(template_path.read_text()))

    annotated = draw_template_regions(img_rgb, template.fields)
    out_path = out_dir / f"{img_path.stem}_template_overlay.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    fields_with_regions = sum(
        1 for f in template.fields if f.label_region or f.value_region
    )
    print(
        f"  {img_path.name}: {fields_with_regions}/{len(template.fields)} campos con región"
        f" → {out_path.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw detection or template overlay on real documents."
    )
    ap.add_argument("images", nargs="+", type=pathlib.Path, help="Image paths.")
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("overlay_output"),
        help="Directory for output images (default: overlay_output/).",
    )
    ap.add_argument(
        "--template",
        type=pathlib.Path,
        default=None,
        metavar="TEMPLATE_JSON",
        help=(
            "Path to a saved template JSON. When given, draws label_region (blue) "
            "and value_region (green) per field directly on the image, without running "
            "the preprocessor or OCR."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.template is not None and not args.template.exists():
        print(f"NOT FOUND: {args.template}", file=sys.stderr)
        sys.exit(1)

    for p in args.images:
        if not p.exists():
            print(f"NOT FOUND: {p}", file=sys.stderr)
            continue
        print(f"Processing {p.name} ...")
        try:
            if args.template is not None:
                process_template_image(p, args.template, args.out_dir)
            else:
                process_image(p, args.out_dir)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
