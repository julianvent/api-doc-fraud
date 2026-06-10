from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.fraud_detection_router import router as fraud_router
from service.logging_config import configure_logging, get_logger


configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load heavy models so the first request doesn't pay the cost."""
    log.info("warming up tampering models")
    from service.tampering import tampering
    tampering.warmup()
    log.info("warming up OCR engine")
    from service.ocr import ocr
    ocr.warmup()
    log.info("ready")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(fraud_router)


@app.get("/health")
async def health():
    return {"msg": "Hola"}
