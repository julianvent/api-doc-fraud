from datetime import date

from fastapi import APIRouter, Form, File, UploadFile
from typing import Annotated

from api.v1.schema.verify import BaseVerifyRequest, BaseVerifyResponse, Identity
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
    identity = Identity(
        full_name=request.full_name,
        date_of_birth=request.date_of_birth,
        gender=request.gender,
    )
    
    return fraud_controller.verify(
        files=request.document_images,
        id=request.id,
        document_type=request.document_type,
        identity=identity
    )


@router.post("/template", response_model=BaseDocumentTemplateResponse)
async def upload_template(
    img: Annotated[UploadFile, File(description="The template image")],
    document_type: Annotated[str, Form()],
    document_name: Annotated[str, Form()],
    country: Annotated[
        str, Form(max_length=5, description="Country code, e.g. MEX, USA, UK")
    ],
    edition: Annotated[date, Form()],
    state: Annotated[str | None, Form()] = None,
):
    template = fraud_controller.upload_template(
        img=img,
        document_type=document_type,
        country=country,
        state=state,
        edition=edition,
        document_name=document_name,
    )

    return template


@router.post("/documents")
async def documents(
    document_images: Annotated[
        list[UploadFile], Form(media_type="multipart/form-data")
    ],
):
    for image in document_images:
        print(image.filename)
