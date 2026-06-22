from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.fraud_detection_router import router as fraud_router
from api.v1.liveness_router import router as liveness_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load heavy models so the first request doesn't pay the cost."""
    print(" > Warming up tampering models...")
    from service.tampering import tampering
    tampering.warmup()
    print(" > Warming up OCR engine...")
    from service.ocr import ocr
    ocr.warmup()
    print(" > Warming up liveness models...")
    from service.liveness import liveness
    liveness.warmup()
    print(" > Warming up active-liveness inference pool...")
    from service import liveness_runtime
    liveness_runtime.warmup()
    print(" > READY!")
    yield
    # Shutdown: drain the inference pool.
    liveness_runtime.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(fraud_router)
app.include_router(liveness_router)


@app.get("/health")
async def health():
    return {"msg": "Hola"}
