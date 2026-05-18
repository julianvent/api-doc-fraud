from model.document_template import DocumentTemplate
from repository.document_template import create_template

from .extractor import extract_template
from .model import TemplateConfig


def upload(
    document_name: str,
    document_type: str,
    img_path: str,
) -> DocumentTemplate:
    """
    Identifica los campos de un documento y los persiste en la base de datos
    Devuelve {personal: [...], document: [...]} con {key, label, type} por campo
    example:
    {
        "personal": [
            {"key": "full_name", "label": "Full Name", "type": "text"},
        ],
        "document": [
            {"key": "document_number", "label": "Document Number", "type": "alphanumeric"},
        ]
    """
    config = TemplateConfig()

    result   = extract_template(img_path, config=config, document_type=document_type)
    personal = result["personal"]
    document = result["document"]

    print(f"[template_ocr] {document_type} → {result['n_fields']} campos "
          f"({len(personal)} personal, {len(document)} document)")

    template = create_template(
        document_name=document_name,
        document_type=document_type,
        img_path=str(img_path),
        fields={"personal": personal, "document": document},
    )

    return template