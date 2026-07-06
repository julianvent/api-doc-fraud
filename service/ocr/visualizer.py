import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

from service.ocr.models import PipelineOutput, TextLine


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "assets", "fonts", "NotoSansDevanagari.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans Devanagari"

GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"


def _to_pixels(bbox: np.ndarray, w: int, h: int) -> np.ndarray:
    pixel_bbox = bbox.astype(np.float32).copy()
    pixel_bbox[:, 0] *= w
    pixel_bbox[:, 1] *= h
    return pixel_bbox


def draw_lines(image_bgr: np.ndarray,
               lines: list[TextLine],
               color: tuple,
               show_text: bool = True) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pixel_bbox = _to_pixels(line.bbox, w, h)
        pts = pixel_bbox.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        if show_text:
            x, y = int(pixel_bbox[0][0]), int(pixel_bbox[0][1]) - 4
            cv2.putText(vis, line.text, (x, max(y, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def visualize_cv2(image_bgr: np.ndarray,
                  output: PipelineOutput,
                  output_path: str = "output/result.jpg"):
    vis = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    color_english = (34, 197, 94)
    color_mrz     = (59, 130, 246)

    for line in output.lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pixel_bbox = _to_pixels(line.bbox, w, h)
        pts   = pixel_bbox.reshape((-1, 1, 2)).astype(np.int32)
        color = color_mrz if _is_mrz_line(line.text) else color_english
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(pixel_bbox[0][0]), int(pixel_bbox[0][1]) - 4
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

    for line in output.lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue

        color = AMBER if _is_mrz_line(line.text) else GREEN

        pixel_bbox = _to_pixels(line.bbox, w, h)
        xs = [pt[0] for pt in pixel_bbox]
        ys = [pt[1] for pt in pixel_bbox]
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


def visualize_template_match(image_bgr: np.ndarray,
                              lines: list[TextLine],
                              template: dict,
                              match_result,
                              output_path: str = "output/template_match.png"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_bgr.shape[:2]

    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pixel_bbox = _to_pixels(line.bbox, w, h)
        xs = [pt[0] for pt in pixel_bbox]
        ys = [pt[1] for pt in pixel_bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.5, edgecolor="#9ca3af", facecolor="none", alpha=0.4
        ))

    assigned_ids = set()
    for value_lines in match_result.field_lines.values():
        for vl in value_lines:
            assigned_ids.add(id(vl))

    for field_def in template.get("fields", []):
        key = field_def.get("key", "")
        label_region = field_def.get("label_region")
        value_region = field_def.get("value_region")

        if label_region:
            x1 = label_region["x1"] * w
            y1 = label_region["y1"] * h
            x2 = label_region["x2"] * w
            y2 = label_region["y2"] * h
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor="#3b82f6", facecolor="none", linestyle="--"
            ))
            ax.text(x1, y1 - 2, f"L:{key}", fontsize=5, color="#3b82f6", va="bottom")

        if value_region:
            x1 = value_region["x1"] * w
            y1 = value_region["y1"] * h
            x2 = value_region["x2"] * w
            y2 = value_region["y2"] * h
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor="#f59e0b", facecolor="none"
            ))
            ax.text(x2, y2 + 2, f"V:{key}", fontsize=5, color="#f59e0b",
                    va="top", ha="right")

    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        if id(line) not in assigned_ids:
            continue
        pixel_bbox = _to_pixels(line.bbox, w, h)
        xs = [pt[0] for pt in pixel_bbox]
        ys = [pt[1] for pt in pixel_bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#22c55e", facecolor="none"
        ))

    legend = [
        patches.Patch(edgecolor="#9ca3af", facecolor="none", label="OCR line"),
        patches.Patch(edgecolor="#3b82f6", facecolor="none", label="label_region"),
        patches.Patch(edgecolor="#f59e0b", facecolor="none", label="value_region"),
        patches.Patch(edgecolor="#22c55e", facecolor="none", label="assigned"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(
        f"{match_result.document_name} | match_score={match_result.match_score:.2f}",
        fontsize=10, pad=10
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved template match: {output_path}")


def _find_line_by_text(lines: list[TextLine], target: str, threshold: int = 60) -> TextLine | None:
    if not target or not lines:
        return None
    from rapidfuzz import fuzz
    target_clean = str(target).strip()
    best_line = None
    best_score = 0
    for l in lines:
        text_clean = l.text.strip()
        score = max(
            fuzz.ratio(text_clean, target_clean),
            fuzz.partial_ratio(text_clean, target_clean),
            fuzz.token_set_ratio(text_clean, target_clean),
        )
        if score > best_score:
            best_score = score
            best_line = l
    return best_line if best_score >= threshold else None


def _bbox_center_pixels(line: TextLine, w: int, h: int) -> tuple[float, float]:
    pixel_bbox = _to_pixels(line.bbox, w, h)
    xs = [pt[0] for pt in pixel_bbox]
    ys = [pt[1] for pt in pixel_bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _line_center_norm(line: TextLine) -> tuple[float, float]:
    xs = [pt[0] for pt in line.bbox]
    ys = [pt[1] for pt in line.bbox]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _find_label_near(lines: list[TextLine], value_line: TextLine, exclude: set) -> TextLine | None:
    vcx, vcy = _line_center_norm(value_line)
    same_row, above = [], []
    for l in lines:
        if l is value_line or id(l) in exclude:
            continue
        lcx, lcy = _line_center_norm(l)
        if abs(lcy - vcy) <= 0.025 and lcx < vcx:
            same_row.append((l, vcx - lcx))
        elif 0 < (vcy - lcy) <= 0.05 and abs(lcx - vcx) <= 0.15:
            above.append((l, vcy - lcy))
    if same_row:
        return min(same_row, key=lambda p: p[1])[0]
    if above:
        return min(above, key=lambda p: p[1])[0]
    return None


def visualize_agent_extraction(image_bgr: np.ndarray,
                                lines: list[TextLine],
                                fields: dict,
                                output_path: str = "output/agent_extraction.png"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_bgr.shape[:2]

    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        pixel_bbox = _to_pixels(line.bbox, w, h)
        xs = [pt[0] for pt in pixel_bbox]
        ys = [pt[1] for pt in pixel_bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0.5, edgecolor="#9ca3af", facecolor="none", alpha=0.4
        ))

    matched = 0
    total = 0
    for key, value in (fields or {}).items():
        if not value:
            continue
        total += 1
        value_line = _find_line_by_text(lines, str(value))
        if value_line is None:
            continue

        pixel_bbox = _to_pixels(value_line.bbox, w, h)
        xs = [pt[0] for pt in pixel_bbox]
        ys = [pt[1] for pt in pixel_bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#22c55e", facecolor="none"
        ))
        ax.text(x2, y2 + 2, key, fontsize=5, color="#22c55e",
                va="top", ha="right")
        matched += 1

    legend = [
        patches.Patch(edgecolor="#9ca3af", facecolor="none", label="OCR line"),
        patches.Patch(edgecolor="#22c55e", facecolor="none", label="Value (agent)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(f"Agent extraction: {matched}/{total} values matched",
                 fontsize=10, pad=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved agent extraction: {output_path}")


def print_report(output: PipelineOutput, agent_result: dict):
    print("\n" + "=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    print(f"Confidence avg : {output.confidence_avg:.2f}")
    print(f"English lines  : {len(output.lines)}")
    print(f"Source         : {output.source}")

    if output.mrz:
        print(f"\n── MRZ ──────────────────────────────────────────────────")
        print(f"  Valid        : {output.mrz.valid}")
        print(f"  Surname      : {output.mrz.surname}")
        print(f"  Given names  : {output.mrz.given_names}")
        print(f"  Country      : {output.mrz.country}")
        print(f"  Birth date   : {output.mrz.date_of_birth}")
        print(f"  Expiry date  : {output.mrz.expiry_date}")
        print(f"  Document number       : {output.mrz.document_number}")
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