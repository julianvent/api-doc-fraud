import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from models import PipelineOutput, TextLine

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "assets", "fonts", "NotoSansDevanagari-Regular.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans Devanagari"

GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"


def draw_lines(image_bgr: np.ndarray,
               lines: list[TextLine],
               color: tuple,
               show_text: bool = True) -> np.ndarray:
    vis = image_bgr.copy()
    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pts = line.bbox.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        if show_text:
            x, y = int(line.bbox[0][0]), int(line.bbox[0][1]) - 4
            cv2.putText(vis, line.text, (x, max(y, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def visualize_cv2(image_bgr: np.ndarray,
                  output: PipelineOutput,
                  output_path: str = "output/result.jpg"):
    vis = image_bgr.copy()
    color_english = (34, 197, 94)
    color_mrz     = (59, 130, 246)

    for line in output.english_lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pts   = line.bbox.reshape((-1, 1, 2)).astype(np.int32)
        color = color_mrz if _is_mrz_line(line.text) else color_english
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(line.bbox[0][0]), int(line.bbox[0][1]) - 4
        cv2.putText(vis, line.text, (x, max(y, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")
    return vis


def visualize_matplotlib(image_bgr: np.ndarray,
                         output: PipelineOutput,
                         output_path: str = "output/result.png"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_bgr.shape[:2]

    for line in output.english_lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue

        color = AMBER if _is_mrz_line(line.text) else GREEN

        xs = [pt[0] for pt in line.bbox]
        ys = [pt[1] for pt in line.bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        ))
        ax.text(x1, y1 - 4, line.text,
                fontsize=6, color=color, va="bottom")

    legend = [
        patches.Patch(edgecolor=GREEN, facecolor="none", label="English text"),
        patches.Patch(edgecolor=AMBER, facecolor="none", label="MRZ line"),
    ]

    if output.mrz:
        mrz_status = "valid" if output.mrz.valid else "invalid"
        legend.append(patches.Patch(
            edgecolor=GREEN if output.mrz.valid else RED,
            facecolor="none",
            label=f"MRZ {mrz_status}"
        ))

    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(f"Source: {output.source} | Confidence avg: {output.confidence_avg:.2f}",
                 fontsize=10, pad=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_report(output: PipelineOutput, agent_result: dict):
    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    print(f"Confidence avg : {output.confidence_avg:.2f}")
    print(f"English lines  : {len(output.english_lines)}")
    print(f"Source         : {output.source}")

    if output.mrz:
        print(f"\n── MRZ ──────────────────────────────────────────────────")
        print(f"  Valid        : {output.mrz.valid}")
        print(f"  Surname      : {output.mrz.surname}")
        print(f"  Given names  : {output.mrz.given_names}")
        print(f"  Country      : {output.mrz.country}")
        print(f"  Birth date   : {output.mrz.birth_date}")
        print(f"  Expiry date  : {output.mrz.expiry_date}")
        print(f"  Number       : {output.mrz.number}")
        print(f"  Sex          : {output.mrz.sex}")

    print(f"\n── AGENT RESULT ─────────────────────────────────────────")
    print(f"  Document type : {agent_result.get('document_type', 'unknown')}")
    print(f"  Verdict       : {agent_result.get('verdict', 'unknown')}")
    print(f"  Confidence    : {agent_result.get('confidence', 'unknown')}")

    fields = agent_result.get("fields", {})
    if fields:
        print(f"\n── EXTRACTED FIELDS ─────────────────────────────────────")
        for k, v in fields.items():
            print(f"  {k:<20} : {v}")

    inconsistencies = agent_result.get("inconsistencies", [])
    if inconsistencies:
        print(f"\n── INCONSISTENCIES ──────────────────────────────────────")
        for inc in inconsistencies:
            print(f"  {inc.get('field')}: {inc.get('description')}")

    print("=" * 60)


def _is_mrz_line(text: str) -> bool:
    import re
    return bool(re.match(r'^[A-Z0-9<]{30,44}$', text.strip().replace(" ", "")))