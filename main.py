from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.fraud_detection_router import router as fraud_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load heavy models so the first request doesn't pay the cost."""
    print(" > Warming up tampering models...")
    from service.tampering import tampering
    tampering.warmup()
    print(" > Warming up OCR engine...")
    from service.ocr import ocr
    ocr.warmup()
    print(" > READY!")
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(fraud_router)


@app.get("/health")
async def health():
    return {"msg": "Hola"}
