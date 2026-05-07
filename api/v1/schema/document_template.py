from pydantic import BaseModel


class BaseDocumentTemplateResponse(BaseModel):
    document_name: str
    document_type: str
    img_path: str
    fields: list[dict]
