from fastapi import APIRouter, Form, File, UploadFile
from typing import Annotated

from api.v1.schema.verify import BaseVerifyRequest, BaseVerifyResponse
from api.v1.schema.document_template import BaseDocumentTemplateResponse
from controller import fraud_detection_controller as fraud_controller

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health():
    return "Enabled"


@router.post("/verify", response_model=BaseVerifyResponse)
async def verify(
    request: Annotated[BaseVerifyRequest, Form(media_type="multipart/form-data")],
):
    return fraud_controller.verify(files=request.document_images, id=request.id, document_type=request.document_type)


@router.post("/template", response_model=BaseDocumentTemplateResponse)
async def upload_template(
    img: Annotated[UploadFile, File(description="The template image")],
    document_type: Annotated[str, Form()],
    document_name: Annotated[str, Form()],
):
    template = fraud_controller.upload_template(
        img=img, document_name=document_name, document_type=document_type
    )

    return template
