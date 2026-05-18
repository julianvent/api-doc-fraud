from pydantic import BaseModel


class TemplateField(BaseModel):
    key  : str
    label: str
    type : str


class BaseDocumentTemplateResponse(BaseModel):
    document_name: str
    document_type: str
    img_path     : str
    fields       : list[TemplateField]