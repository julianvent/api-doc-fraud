import json
from pathlib import Path
from typing import Any, Optional

from model.document_template import DocumentTemplate
from repository.document_template import create_template

from .extractor import extract_template
from .model import TemplateConfig


def _preclassify_image(img_path: str):
    """OCR + preclassifier over the template image to recover doc_family / mrz_type / country_iso."""
    try:
        from service.ocr.engine import PaddleOCRAdapter, load_image
        from service.ocr.models import Config as OcrConfig
        from service.ocr.preclassifier import classify

        cfg    = OcrConfig()
        engine = PaddleOCRAdapter(cfg)
        image  = load_image(Path(img_path))
        lines  = engine.extract(image)
        return classify(image, lines)
    except Exception as e:
        print(f"  [template_ocr] preclassifier failed: {type(e).__name__}: {e}")
        return None


def _detect_qr(img_path: str) -> dict:
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return {"present": False, "signed": False}
        detector       = cv2.QRCodeDetector()
        data, _, _     = detector.detectAndDecode(img)
        if not data:
            return {"present": False, "signed": False}
        return {"present": True, "signed": False, "format": "qr", "fields_map": {}}
    except Exception as e:
        print(f"  [template_ocr] qr detect failed: {type(e).__name__}: {e}")
        return {"present": False, "signed": False}


def _build_v2_dict(
    document_type: str,
    document_name: str,
    img_path: str,
    country: Optional[str],
    personal: list[dict],
    document: list[dict],
    fingerprint: dict,
    validators: dict,
    qr_config: dict,
    preclass,
) -> dict:
    flat_fields = (
        [{**f, "category": "personal"} for f in personal] +
        [{**f, "category": "document"} for f in document]
    )

    country_iso = (
        country.upper() if country and len(country) == 3 else None
    ) or (preclass.country_iso if preclass else None)

    return {
        "schema_version": 2,
        "document_type":  document_type,
        "document_name":  document_name,
        "country":        country,
        "country_iso":    country_iso,
        "doc_family":     preclass.doc_family if preclass else None,
        "mrz_type":       preclass.mrz_type if preclass else None,
        "img_path":       str(img_path),
        "fingerprint":    fingerprint,
        "fields":         flat_fields,
        "field_rules":    validators,
        "qr_config":      qr_config,
    }


def _write_v2_template(template_dict: dict, document_type: str) -> None:
    from service.ocr.models import Config as OcrConfig

    dir_path = Path(OcrConfig().templates_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / f"{document_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template_dict, f, ensure_ascii=False, indent=2)
    print(f"  [template_ocr] wrote v2 JSON → {out_path}")


def _index_in_qdrant(template_id: Any, v2_dict: dict, preclass) -> bool:
    from service.ocr import matching
    from service.ocr.models import Config as OcrConfig
    from service.template_ocr.schema import Fingerprint

    cfg = OcrConfig()
    if cfg.disable_vector_match or not matching.is_available():
        print("  [template_ocr] qdrant unavailable or disabled — skipping upsert")
        return False

    fp_data = v2_dict.get("fingerprint") or {}
    fp_obj  = Fingerprint(
        layout_desc = fp_data.get("layout_desc"),
        anchors     = fp_data.get("anchors") or [],
    )
    text = matching.serialize_for_template(fp_obj, preclass)
    if not text:
        print("  [template_ocr] empty fingerprint text — skipping upsert")
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
        cfg.qdrant_url,
        cfg.qdrant_collection,
        point_id   = str(template_id),
        vector     = vector,
        payload    = payload,
        vector_size = len(vector),
    )


def upload(
    document_name: str,
    document_type: str,
    img_path: str,
    country: str | None = None,
) -> DocumentTemplate:
    """
    Genera template enriquecida (v2): campos + fingerprint + validators + qr_config.
    Persiste:
      - JSON v2 en service/ocr/templates/{document_type}.json (source for verify path)
      - Fila en DB (shape v1 por ahora; las columnas v2 se añadirán cuando se haga la
        migración Alembic — ver plan Step 8, fuera del alcance actual de este refactor)
      - Punto en Qdrant si la collection está disponible
    """
    config = TemplateConfig()

    result      = extract_template(img_path, config=config, document_type=document_type, country=country)
    personal    = result["personal"]
    document    = result["document"]
    fingerprint = result.get("fingerprint", {})
    validators  = result.get("validators", {})

    print(
        f"[template_ocr] {document_type} → {result['n_fields']} campos "
        f"({len(personal)} personal, {len(document)} document); "
        f"fingerprint.anchors={len((fingerprint or {}).get('anchors') or [])}; "
        f"validators={len(validators or {})}"
    )

    preclass  = _preclassify_image(img_path)
    qr_config = _detect_qr(img_path)

    v2_dict = _build_v2_dict(
        document_type = document_type,
        document_name = document_name,
        img_path      = img_path,
        country       = country,
        personal      = personal,
        document      = document,
        fingerprint   = fingerprint,
        validators    = validators,
        qr_config     = qr_config,
        preclass      = preclass,
    )

    _write_v2_template(v2_dict, document_type)

    # TODO(step 8 — out of current refactor scope): once model/document_template.py
    # and repository/document_template.create_template gain the v2 columns
    # (country_iso, doc_family, mrz_type, fingerprint, field_rules, qr_config,
    # schema_version, embedding_id), pass them here as kwargs. Today the disk JSON
    # and Qdrant carry the enriched data; the DB row keeps the v1 shape.
    template = create_template(
        document_type = document_type,
        country       = country,
        document_name = document_name,
        img_path      = str(img_path),
        fields        = {"personal": personal, "document": document},
    )

    template_id = getattr(template, "id", None)
    if template_id is not None:
        _index_in_qdrant(template_id, v2_dict, preclass)

    return template
