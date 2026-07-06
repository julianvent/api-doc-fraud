"""Integration tests for the liveness API (active WebSocket session)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LIVENESS_LOG_LEVEL", "ERROR")

import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1.liveness_router import router
from service import liveness_runtime
from service.liveness import liveness

_FACE_GLOB = str(
    Path(__file__).resolve().parent
    / "service" / "liveness" / "dataset" / "bona_fide" / "WIN_*.jpg"
)


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    liveness.warmup()
    liveness_runtime.warmup()
    with TestClient(app) as c:
        yield c
    liveness_runtime.shutdown()


@pytest.fixture(scope="module")
def frame_bytes():
    paths = sorted(glob.glob(_FACE_GLOB))
    if not paths:
        pytest.skip("no dataset face images")
    img = cv2.resize(cv2.imread(paths[0]), (1280, 720))
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    r = client.get("/v1/liveness/health")
    assert r.status_code == 200
    assert r.json() == "Enabled"


def test_client_served(client):
    assert client.get("/v1/liveness/client").status_code == 200


def test_active_ws_reaches_challenges(client, frame_bytes):
    """Active: session_started -> preflight -> challenge_active.

    Static frames can't complete motion challenges, so we only assert
    the protocol advances past preflight into the challenge phase.
    """
    with client.websocket_connect("/v1/liveness/session") as ws:
        started = ws.receive_json()
        assert started["type"] == "session_started"
        assert len(started["challenges"]) >= 2

        phases = set()
        for _ in range(60):
            ws.send_bytes(frame_bytes)
            msg = ws.receive_json()
            if msg["type"] == "snapshot":
                phases.add(msg["phase"])
            elif msg["type"] == "result":
                break

        assert "preflight" in phases
        assert "challenge_active" in phases


def test_busy_rejects_second_session(client, frame_bytes):
    """The single-slot registry rejects a concurrent connection."""
    with client.websocket_connect("/v1/liveness/session") as ws1:
        ws1.receive_json()  # session_started — slot now held
        ws1.send_bytes(frame_bytes)
        ws1.receive_json()
        with client.websocket_connect("/v1/liveness/session") as ws2:
            msg = ws2.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "busy"
