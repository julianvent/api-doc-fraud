"""
Idempotent script to upgrade v1 template JSONs on disk to enriched v2.

Usage:
  python -m service.template_ocr.migrate_v1_to_v2                    # all templates
  python -m service.template_ocr.migrate_v1_to_v2 passport visa      # specific types
  python -m service.template_ocr.migrate_v1_to_v2 --dry-run          # show plan, no writes

For each template:
  - Skips if schema_version >= 2 (already migrated).
  - Loads the v1 JSON, normalizes to v2 via the schema adapter.
  - If img_path exists, runs OCR + preclassifier (doc_family / mrz_type / country_iso).
  - If img_path exists AND --regen-vlm is set, calls the VLM template extractor
    to refresh fingerprint and validators (requires Ollama running).
  - Detects QR via OpenCV.
  - Writes the enriched JSON back in place.
  - If Qdrant is available, embeds the fingerprint and upserts the point.

This script only touches files inside service/ocr/templates/ and Qdrant.
It does NOT modify the database.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_img_path(raw_img_path: Optional[str], repo_root: Path) -> Optional[Path]:
    if not raw_img_path:
        return None
    p = Path(raw_img_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = repo_root / raw_img_path
    if candidate.exists():
        return candidate
    return None


def _preclassify(img_path: Path):
    try:
        from service.ocr.engine import load_image
        from service.ocr.models import Config
        from service.ocr.ocr import _get_engine
        from service.ocr.preclassifier import classify

        cfg    = Config()
        engine = _get_engine(cfg)
        image  = load_image(img_path)
        lines  = engine.extract(image)
        return classify(image, lines)
    except Exception as e:
        print(f"    preclassifier failed: {type(e).__name__}: {e}")
        return None


def _detect_qr(img_path: Path) -> dict:
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return {"present": False, "signed": False}
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if not data:
            return {"present": False, "signed": False}
        return {"present": True, "signed": False, "format": "qr", "fields_map": {}}
    except Exception:
        return {"present": False, "signed": False}


def _regen_via_vlm(img_path: Path, document_type: str, country: Optional[str]) -> tuple[dict, dict]:
    """Returns (fingerprint, validators). Empty dicts on failure."""
    try:
        from .extractor import extract_template
        from .model import TemplateConfig

        result      = extract_template(str(img_path), config=TemplateConfig(),
                                       document_type=document_type, country=country)
        fingerprint = result.get("fingerprint") or {}
        validators  = result.get("validators") or {}
        return fingerprint, validators
    except Exception as e:
        print(f"    VLM regen failed: {type(e).__name__}: {e}")
        return {}, {}


def _index_in_qdrant(template_id: str, v2_dict: dict, preclass) -> bool:
    try:
        from service.ocr import matching
        from service.ocr.models import Config
        from .schema import Fingerprint

        cfg = Config()
        if cfg.disable_vector_match or not matching.is_available():
            return False

        fp_data = v2_dict.get("fingerprint") or {}
        fp_obj  = Fingerprint(
            layout_desc = fp_data.get("layout_desc"),
            anchors     = fp_data.get("anchors") or [],
        )
        text = matching.serialize_for_template(fp_obj, preclass)
        if not text:
            return False

        vector = matching.embed(text, cfg.embedding_url, cfg.embedding_model)
        if vector is None:
            return False

        payload = {
            "template_id":   template_id,
            "document_type": v2_dict.get("document_type"),
            "country_iso":   v2_dict.get("country_iso"),
            "doc_family":    v2_dict.get("doc_family"),
            "mrz_type":      v2_dict.get("mrz_type"),
        }
        return matching.upsert(
            cfg.qdrant_url, cfg.qdrant_collection,
            point_id    = template_id,
            vector      = vector,
            payload     = payload,
            vector_size = len(vector),
        )
    except Exception as e:
        print(f"    qdrant index failed: {type(e).__name__}: {e}")
        return False


def upgrade_one(json_path: Path, repo_root: Path, regen_vlm: bool, dry_run: bool) -> bool:
    print(f"\n--- {json_path.name}")
    raw = _load_json(json_path)
    if int(raw.get("schema_version", 1)) >= 2:
        print("    already v2, skipping")
        return False

    from .schema import load_template

    normalized = load_template(raw).model_dump(exclude_none=True)
    normalized["schema_version"] = 2
    if "fingerprint" not in normalized:
        normalized["fingerprint"] = {"layout_desc": None, "anchors": []}
    if "field_rules" not in normalized:
        normalized["field_rules"] = {}
    if "qr_config" not in normalized:
        normalized["qr_config"] = {"present": False, "signed": False}

    img_path = _resolve_img_path(raw.get("img_path"), repo_root)
    preclass = None
    if img_path is not None:
        print(f"    img_path: {img_path}")
        preclass = _preclassify(img_path)
        if preclass:
            normalized["doc_family"]  = preclass.doc_family
            normalized["mrz_type"]    = preclass.mrz_type
            normalized["country_iso"] = preclass.country_iso or normalized.get("country_iso")
            print(f"    preclass: family={preclass.doc_family} mrz={preclass.mrz_type} country={preclass.country_iso}")
        normalized["qr_config"] = _detect_qr(img_path)

        if regen_vlm:
            print("    regenerating fingerprint+validators via VLM...")
            fingerprint, validators = _regen_via_vlm(
                img_path, normalized.get("document_type", "generic"), normalized.get("country")
            )
            if fingerprint:
                normalized["fingerprint"] = fingerprint
            if validators:
                normalized["field_rules"] = validators
    else:
        print(f"    img_path not resolvable; v1 normalized without preclass/fingerprint")

    if dry_run:
        print("    [dry-run] would write:", json.dumps(
            {k: normalized.get(k) for k in
             ("schema_version", "doc_family", "mrz_type", "country_iso", "field_rules")},
            ensure_ascii=False,
        ))
        return False

    _write_json(json_path, normalized)
    print(f"    wrote v2 -> {json_path}")

    template_id = f"{normalized.get('document_type', 'unknown')}_{normalized.get('country_iso') or normalized.get('country') or 'na'}"
    if _index_in_qdrant(template_id, normalized, preclass):
        print(f"    qdrant upsert ok (id={template_id})")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Upgrade v1 template JSONs to v2.")
    parser.add_argument("doc_types", nargs="*",
                        help="Specific document_types to migrate (e.g. passport visa). Empty = all.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing.")
    parser.add_argument("--regen-vlm", action="store_true",
                        help="Also re-run the VLM template extractor to refresh fingerprint+validators.")
    args = parser.parse_args()

    repo_root     = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "service" / "ocr" / "templates"

    if not templates_dir.exists():
        print(f"templates dir not found: {templates_dir}", file=sys.stderr)
        return 1

    if args.doc_types:
        targets = [templates_dir / f"{t}.json" for t in args.doc_types]
        targets = [t for t in targets if t.exists()]
    else:
        targets = sorted(templates_dir.glob("*.json"))

    print(f"Found {len(targets)} template(s) under {templates_dir}")
    updated = 0
    for path in targets:
        if upgrade_one(path, repo_root, regen_vlm=args.regen_vlm, dry_run=args.dry_run):
            updated += 1
    print(f"\nDone. Updated {updated}/{len(targets)} templates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
