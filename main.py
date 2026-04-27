from fastapi import FastAPI
from api.v1.fraud_detection_router import router as fraud_router

app = FastAPI()
app.include_router(fraud_router)


@app.get("/health")
async def health():
    return {"msg": "Hola"}
