from pydantic import BaseModel


class BaseVerifyRequest(BaseModel):
    pass


class BaseVerifyResponse(BaseModel):
    tampering_score: float
    flags: list[str]
    confidence: float
