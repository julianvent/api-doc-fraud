from db.session import Session
from model.document_template import DocumentTemplate


def create_template(
    document_name: str, document_type: str, fields: list[dict], img_path: str, country: str | None = None #anchors: list[dict], img_path: str
) -> DocumentTemplate:
    template = DocumentTemplate(
        document_name=document_name,
        document_type=document_type,
        country=country,
        img_path=img_path,
        fields=fields,
        #anchors=anchors,
    )
    # with Session.begin() as s:
    #     s.add(template)
    return template
