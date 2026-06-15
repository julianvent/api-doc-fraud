import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Allow to call the API
_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000")
allow_origins = [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fraud_router)


@app.get("/health")
async def health():
    return {"msg": "Hola"}
