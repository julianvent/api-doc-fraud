from pydantic import BaseModel


class TemplateField(BaseModel):
    key  : str
    label: str
    type : str


class BaseDocumentTemplateResponse(BaseModel):
    document_type: str
    country      : str | None = None
    document_name: str
    img_path     : str
    fields       : dict[str, list[TemplateField]]  # {"personal": [...], "document": [...]}