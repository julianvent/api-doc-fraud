"""Liveness API routes — camera only """
from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.v1.schema.liveness import LivenessVerdict
from api.v1.schema.liveness_ws import (
    ChallengeInfo,
    ErrorMsg,
    OvalSchema,
    ResultMsg,
    SessionStartedMsg,
    SnapshotMsg,
)
from service import liveness_runtime

from service.liveness.active import DEFAULT_POOL
from service.liveness.domain.verdict import Verdict as LivenessModuleVerdict
from service.liveness.runtime import ActiveSessionDriver

# A session asks for CENTER_FACE + every challenge in the pool.
_N_CHALLENGES = len(DEFAULT_POOL) + 1
_STATIC = Path(__file__).resolve().parents[2] / "static"

router = APIRouter(prefix="/v1")


@router.get("/liveness/health")
async def liveness_health():
    return "Enabled"


@router.get("/liveness/client")
async def liveness_client():
    """Liveness browser test client (camera + challenges; PAD runs inside)."""
    return FileResponse(_STATIC / "liveness_client.html", media_type="text/html")


@router.websocket("/liveness/session")
async def liveness_session(ws: WebSocket):
    config = liveness_runtime.get_config()
    driver = ActiveSessionDriver(
        liveness_runtime.get_engine(), n_challenges=_N_CHALLENGES, config=config
    )
    await _run_ws(
        ws, driver,
        started=lambda d: _active_started_msg(d, config),
        snapshot=_active_snapshot_msg,
        result=_result_msg,
    )


# ── WebSocket loop (drop-oldest) ────────────────────────────────────────────────

async def _run_ws(ws: WebSocket, driver, *, started, snapshot, result) -> None:
    """Session loop: accept, acquire a slot, stream frames with
    drop-oldest, emit snapshots, emit the final result. The driver
    exposes `.session_id`, `.start()`, `.process_frame()`, and an update
    with `.result`."""
    await ws.accept()
    registry = liveness_runtime.get_registry()
    if not registry.try_acquire(driver.session_id):
        await ws.send_json(
            ErrorMsg(code="busy", message="Liveness is busy. Try again shortly.").model_dump()
        )
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    got_frame = asyncio.Event()
    latest: dict = {"frame": None}

    async def _receiver():
        try:
            while not stop.is_set():
                latest["frame"] = await ws.receive_bytes()
                got_frame.set()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            stop.set()
            got_frame.set()

    recv_task = asyncio.create_task(_receiver())
    try:
        driver.start(loop.time())
        await ws.send_json(started(driver).model_dump())

        while not stop.is_set():
            await got_frame.wait()
            got_frame.clear()
            if stop.is_set():
                break
            frame_bytes = latest["frame"]
            latest["frame"] = None
            if not frame_bytes:
                continue
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json(
                    ErrorMsg(code="bad_frame", message="Undecodable frame.").model_dump()
                )
                continue

            update = await driver.process_frame(frame, loop.time())
            await ws.send_json(snapshot(update).model_dump())
            if update.result is not None:
                await ws.send_json(result(driver.session_id, update.result).model_dump())
                break
    finally:
        stop.set()
        recv_task.cancel()
        registry.release(driver.session_id)
        try:
            await ws.close()
        except RuntimeError:
            pass


# ── message mapping: active ─────────────────────────────────────────────────────

def _active_started_msg(driver, config) -> SessionStartedMsg:
    o = config.oval
    return SessionStartedMsg(
        session_id=driver.session_id,
        challenges=[c.value for c in driver.selected_challenges],
        oval=OvalSchema(cx=o.cx, cy=o.cy, rx=o.rx, ry=o.ry),
        send_fps=config.send_fps,
        send_width=config.send_width,
        send_height=config.send_height,
        max_session_seconds=config.session_timeout_s,
    )


def _active_snapshot_msg(update) -> SnapshotMsg:
    snap = update.snapshot
    challenge = None
    if snap.current_challenge is not None:
        challenge = ChallengeInfo(
            index=snap.challenge_index,
            total=snap.n_challenges,
            type=snap.current_challenge.type.value,
            prompt=snap.current_challenge.prompt,
        )
    preflight_progress = 0.0
    if snap.preflight_duration_s > 0:
        preflight_progress = min(1.0, snap.preflight_elapsed_s / snap.preflight_duration_s)
    return SnapshotMsg(
        phase=snap.phase.value,
        challenge=challenge,
        challenge_phase=snap.challenge_phase,
        time_remaining_s=round(snap.time_remaining_s, 2),
        in_oval=update.in_oval,
        wrong_direction=snap.wrong_direction,
        occlusion_hint=update.occlusion_hint,
        preflight_progress=round(preflight_progress, 2),
    )


# ── result mapping ──────────────────────────────────────────────────────────────

def _result_msg(session_id: str, result) -> ResultMsg:
    verdict = (
        LivenessVerdict.ACCEPT
        if result.verdict == LivenessModuleVerdict.ACCEPT
        else LivenessVerdict.REJECT
    )
    return ResultMsg(
        session_id=session_id,
        verdict=verdict,
        reasons=list(result.reasons),
        message=result.message,
    )
