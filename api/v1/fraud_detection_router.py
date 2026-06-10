from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from api.v1.schema.document_template import TemplateDetail, TemplateSummary
from api.v1.schema.template_confirm import ConfirmTemplateRequest
from api.v1.schema.template_generate import GenerateResponse
from api.v1.schema.verify import BaseVerifyRequest, BaseVerifyResponse
from controller import fraud_detection_controller as fraud_controller
from controller import template_controller


router = APIRouter(prefix="/v1")


@router.get("/health")
async def health():
    return "Enabled"


@router.post("/verify", response_model=BaseVerifyResponse)
async def verify(
    request: Annotated[BaseVerifyRequest, Form(media_type="multipart/form-data")],
):
    return fraud_controller.verify(
        files=request.document_images,
        id=request.id,
        document_type=request.document_type,
    )


# ─────────────────────────────────────────────────────────────── templates


@router.get("/templates", response_model=list[TemplateSummary])
async def list_templates(
    document_type: Annotated[Optional[str], Query()] = None,
    country: Annotated[Optional[str], Query()] = None,
):
    return template_controller.list_templates(document_type=document_type, country=country)


@router.get("/templates/{template_id}", response_model=TemplateDetail)
async def get_template(template_id: str):
    return template_controller.get_template(template_id)


@router.post("/templates/generate", response_model=GenerateResponse)
async def generate_template(
    image: Annotated[UploadFile, File(description="Sample document image")],
    mode: Annotated[str, Form(description="auto | manual")],
    expected_fields: Annotated[
        Optional[str],
        Form(description='JSON array of {key,label,type} — required when mode=manual'),
    ] = None,
):
    parsed_expected = None
    if expected_fields:
        import json
        try:
            parsed_expected = json.loads(expected_fields)
            if not isinstance(parsed_expected, list):
                raise ValueError("expected_fields must be a JSON array")
        except (ValueError, json.JSONDecodeError) as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail=f"invalid expected_fields: {e}")
    return template_controller.generate_template(
        image=image,
        mode=mode,
        expected_fields=parsed_expected,
    )


@router.post(
    "/templates/confirm",
    response_model=TemplateDetail,
    status_code=status.HTTP_201_CREATED,
)
async def confirm_template(req: ConfirmTemplateRequest):
    return template_controller.confirm_template(req)
