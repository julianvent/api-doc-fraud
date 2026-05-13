from pathlib import Path

from model.document_template import DocumentTemplate
from repository.document_template import create_template

from .extractor import extract_template
from .model import TemplateConfig
from .visualizer import visualize


def upload(
    document_name: str,
    document_type: str,
    img_path: str,
    include_type: bool = False,
) -> DocumentTemplate:
    """
    Genera la plantilla de un tipo de documento y la persiste en la base de datos.
    """
    config = TemplateConfig()

    # Extraer campos con dots.ocr + LLM
    result = extract_template(img_path, config=config, document_type=document_type, include_type=include_type, explicit=False) #, expand_x=True) # False para x's del value fijas
    fields = result["fields"]
    #anchors = result.get("anchors", [])

    print(f"[template_ocr] {document_type} → {result['n_fields']} fields extraídos")
    print(fields)

    # Guardar visualizacion
    output_dir  = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(img_path).stem}_template_viz.png"
    visualize(img_path, output_path=output_path, config=config, fields=fields)

    # Persistir en base de datos
    template = create_template(
        document_name=document_name,
        document_type=document_type,
        img_path=str(img_path),
        fields=fields,
        #anchors=anchors,
    )

    return template