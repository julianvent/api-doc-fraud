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
    Identifica los campos de un documento y los persiste en la base de datos.
    Devuelve {key, label, type} por campo, sin coordenadas.
    example:
    {
        "key": "field_1",
        "label": "Nombre",
        "type": "text"
    }
    """
    config = TemplateConfig()

    result = extract_template(img_path, config=config, document_type=document_type)
    fields = result["fields"]

    print(f"[template_ocr] {document_type} → {result['n_fields']} campos identificados")
    print(fields)

    template = create_template(
        document_name=document_name,
        document_type=document_type,
        img_path=str(img_path),
        fields=fields,
    )

    return template