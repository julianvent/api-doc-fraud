r"""Minimal HTTPS launcher for the liveness service alone."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi import FastAPI

from api.v1.liveness_router import router as liveness_router
from service import liveness_runtime
from service.liveness import liveness


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the passive facade + the active inference pool once.
    liveness.warmup()
    liveness_runtime.warmup()
    yield
    liveness_runtime.shutdown()


app = FastAPI(title="Liveness (standalone)", lifespan=lifespan)
app.include_router(liveness_router)
