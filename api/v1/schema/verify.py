from pydantic import BaseModel
from fastapi import UploadFile


class BaseVerifyRequest(BaseModel):
    document_images: list[UploadFile]
    id: str


class BaseVerifyResponse(BaseModel):
    tampering_score: float
    flags: list[str]
    confidence: float
