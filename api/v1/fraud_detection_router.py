from fastapi import APIRouter, File, UploadFile, Form
from typing import Annotated

from schema.verify import BaseVerifyResponse
from controller import preprocessor

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health():
    return "Enabled"


@router.post("/verify/", response_model=BaseVerifyResponse)
async def verify(document_images: list[UploadFile], id: Annotated[str, Form()]):
    preprocessor.process_files(files=document_images, id=id)

    mock_response = BaseVerifyResponse(
        tampering_score=0.9, flags=["dob_incositency"], confidence=0.8
    )
    return mock_response
