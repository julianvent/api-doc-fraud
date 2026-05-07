from model.document_template import DocumentTemplate
from repository.document_template import create_template


def upload(
    document_name: str,
    document_type: str,
    img_path: str,
) -> DocumentTemplate:
    # perform ocr for extracting fields
    fields: list[dict] = [{"name": "Full name", "bbox": {1, 2, 3, 4}}]

    """ Fields schema example 
    
    fields = {
        "date_of_birth": {
            "label": "Date of birth",
            "label_region": [x1, y1, x2, y2],
        },
        "given_name": {
            "label": "Given name",
            "label_region": [x1, y1, x2, y2]
        },
        ...
    }
    
    """

    # log output for verification
    print(fields)

    # commit to database
    template = create_template(
        document_name=document_name,
        document_type=document_type,
        img_path=img_path,
        fields=fields,
    )

    return template
