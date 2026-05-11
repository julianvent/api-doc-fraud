import argparse
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from .extractor import extract_template, _load_image
from .model import TemplateConfig


COLOR_LABEL = (0, 200, 150)
COLOR_VALUE = (0, 102, 204)
COLOR_TEXT  = (255, 128, 0)
BOX_ALPHA   = 0
LINE_WIDTH  = 2


def _denormalize(region: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    return (
        int(region["x1"] * img_w),
        int(region["y1"] * img_h),
        int(region["x2"] * img_w),
        int(region["y2"] * img_h),
    )


def _draw_box(draw, overlay, box, color, label):
    x1, y1, x2, y2 = box
    ImageDraw.Draw(overlay).rectangle([x1, y1, x2, y2], fill=(*color, BOX_ALPHA))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

    font_size = max(10, min(14, (y2 - y1) // 2))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x1 + 3, y1 + 2), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1 + 3, y1 + 2), label, fill=COLOR_TEXT, font=font)


def _draw_legend(draw, img_w):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    x, y = img_w - 160, 10
    for color, text in [(COLOR_LABEL, "label_region"), (COLOR_VALUE, "value_region")]:
        draw.rectangle([x, y + 2, x + 16, y + 14], fill=color)
        draw.text((x + 22, y), text, fill=(30, 30, 30), font=font)
        y += 22


def visualize(
    img_path: str | Path,
    output_path: str | Path | None = None,
    config: "TemplateConfig | None" = None,
) -> Path:
    if config is None:
        config = TemplateConfig()

    img_path = Path(img_path)

    if output_path is None:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{img_path.stem}_template_viz.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[visualizer] Extrayendo template de: {img_path.name}")
    result = extract_template(img_path, config=config)
    fields = result["fields"]

    print(f"[visualizer] {result['n_fields']} fields detectados\n")
    for f in fields:
        print(f"  [{f['key']}]")
        print(f"    label       : {f['label']}")
        print(f"    label_region: {f['label_region']}")
        print(f"    value_region: {f['value_region']}")
        print()

    img_array = _load_image(img_path)
    base_img  = PILImage.fromarray(img_array).convert("RGBA")
    img_w, img_h = base_img.size
    overlay  = PILImage.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(base_img)

    for field in fields:
        key = field["key"]
        lb  = _denormalize(field["label_region"], img_w, img_h)
        vb  = _denormalize(field["value_region"], img_w, img_h)
        _draw_box(draw, overlay, lb, COLOR_LABEL, f"L: {key}")
        _draw_box(draw, overlay, vb, COLOR_VALUE, f"V: {key}")

    combined = PILImage.alpha_composite(base_img, overlay).convert("RGB")
    _draw_legend(ImageDraw.Draw(combined), img_w)
    combined.save(output_path)

    print(f"[visualizer] Guardado en: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",  required=True)
    parser.add_argument("--out",  default=None)
    args = parser.parse_args()

    visualize(img_path=args.img, output_path=args.out)