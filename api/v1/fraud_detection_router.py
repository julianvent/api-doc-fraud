import asyncio
import json as _json
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from api.v1.schema.document_template import TemplateDetail, TemplateSummary
from api.v1.schema.template_confirm import ConfirmTemplateRequest
from api.v1.schema.template_generate import GenerateResponse
from api.v1.schema.verify import BaseVerifyRequest, BaseVerifyResponse
from controller import fraud_detection_controller as fraud_controller
from controller import template_controller
from service.template_ocr import scan_cache as _scan_cache


router = APIRouter(prefix="/v1")


async def _sse_stream(image: UploadFile, mode: str) -> None:
    """SSE wrapper for generate modes that may take several seconds (dots, manual)."""
    # Send headers + first bytes immediately so every proxy/browser timer resets.
    yield ": keepalive\n\n"

    fut = asyncio.ensure_future(
        asyncio.to_thread(
            template_controller.generate_template,
            image=image,
            mode=mode,
        )
    )

    while not fut.done():
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=5.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except Exception:
            break

    try:
        result: GenerateResponse = fut.result()
        yield f"data: {result.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except HTTPException as e:
        yield f"event: error\ndata: {_json.dumps({'detail': e.detail})}\n\n"
    except Exception as e:
        yield f"event: error\ndata: {_json.dumps({'detail': str(e)})}\n\n"


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


@router.post("/templates/generate")
async def generate_template(
    image: Annotated[UploadFile, File(description="Sample document image")],
    mode: Annotated[str, Form(description="auto | manual | dots")],
):
    if mode in ("dots", "manual"):
        return StreamingResponse(
            _sse_stream(image=image, mode=mode),
            media_type="text/event-stream",
        )

    return await asyncio.to_thread(
        template_controller.generate_template,
        image=image,
        mode=mode,
    )


@router.post(
    "/templates/confirm",
    response_model=TemplateDetail,
    status_code=status.HTTP_201_CREATED,
)
async def confirm_template(req: ConfirmTemplateRequest):
    return template_controller.confirm_template(req)


@router.get("/templates/session/{generate_id}/image")
async def get_session_image(generate_id: str):
    """Serve the preprocessed image for an active generate session.

    Available from the moment generate returns until confirm is called
    (confirm deletes the scan-cache entry). Returns 404 for expired,
    already-confirmed, or mode=auto sessions (auto does not save a
    preprocessed image).
    """
    path = _scan_cache.path_for_preprocessed(generate_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="no preprocessed image for this generate_id "
                   "(expired, already confirmed, or mode=auto)",
        )
    return FileResponse(path, media_type="image/png")
