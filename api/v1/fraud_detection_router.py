from fastapi import APIRouter, Form
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
    return fraud_controller.verify(files=request.document_images, id=request.id)
