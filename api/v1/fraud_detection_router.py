from fastapi import APIRouter, UploadFile, Form
from typing import Annotated

from api.v1.schema.verify import BaseVerifyRequest, BaseVerifyResponse
from controller import fraud_detection_controller as fraud_controller

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health():
    return "Enabled"


@router.post("/verify/", response_model=BaseVerifyResponse)
async def verify(
    request: Annotated[BaseVerifyRequest, Form(media_type="multipart/form-data")],
):
    fraud_controller.process_files(files=request.document_images, id=request.id)

    mock_response = BaseVerifyResponse(
        tampering_score=0.9, flags=["dob_incositency"], confidence=0.8
    )
    return mock_response
