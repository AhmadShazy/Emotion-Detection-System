"""
routers/stream.py

WebSocket endpoint: ws://localhost:8000/ws/stream

Flow:
    1. Client connects (browser opens WebSocket)
    2. Server performs a sounddevice availability check BEFORE accepting. (LS1)
       If the server has no microphone, a clear error is sent to the browser
       and the connection is closed — no orphaned orchestrator threads.
    3. Server accepts + starts run_live_streaming_session_ws() in a daemon thread,
       passing the optional session_id (LS3) so streaming can inherit conversation
       history from prior text/voice sessions.
    4. Each speech turn calls on_payload(dict), bridged thread-safely to send_json().
    5. The async loop polls for client messages / WebSocketDisconnect.
    6. On disconnect, stop_event is set → all workers shut down gracefully.
"""

import sys
import os
import asyncio
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, Query
from fastapi.websockets import WebSocket, WebSocketDisconnect
from typing import Optional

from src.streaming.live_orchestrator import run_live_streaming_session_ws

router = APIRouter()


def _check_server_audio() -> tuple:
    """
    Verifies that the server's default audio input device is available.

    Returns:
        (True, "")        — microphone is accessible
        (False, <reason>) — microphone unavailable, reason is user-facing

    LS1 fix: Without this guard a missing/muted server mic causes the
    AudioStreamer to fail silently after WS is accepted, leaving the
    client connected indefinitely with no results.
    """
    try:
        import sounddevice as sd
        sd.check_input_settings(samplerate=16000, channels=1, dtype="float32")
        return True, ""
    except Exception as exc:
        return False, (
            f"Server microphone unavailable: {exc}. "
            "Live Stream requires the server to run on a machine with a working "
            "microphone (device 0). Check your system audio input settings."
        )


@router.websocket("/ws/stream")
async def live_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default=None),
):
    """
    WebSocket handler for Option 4 — Live Multimodal Streaming.

    Query params:
        session_id (optional) — pass a prior session_id to inherit
                                 conversation history. (LS3 fix)
    """
    # ── LS1: Pre-flight microphone check ─────────────────────────────────────
    # Done before accept() so we can still reject cleanly.
    # FastAPI requires accept() before send, so we accept → send error → close.
    mic_ok, mic_error = _check_server_audio()
    if not mic_ok:
        await websocket.accept()
        await websocket.send_json({
            "type":    "error",
            "code":    "MIC_UNAVAILABLE",
            "message": mic_error,
        })
        await websocket.close(code=1011)
        print(f"[Stream] \u274c WS rejected \u2014 server mic unavailable.")
        return

    await websocket.accept()

    # ── Shared state between async loop and orchestrator thread ──────────────
    stop_event = threading.Event()
    loop       = asyncio.get_event_loop()

    # ── Thread-safe payload sender ────────────────────────────────────────────
    def on_payload(payload: dict):
        """Called from the background thread — bridges to async send_json()."""
        if stop_event.is_set():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                websocket.send_json(payload),
                loop,
            )
            future.result(timeout=5)
        except Exception:
            stop_event.set()

    # ── Launch orchestrator in daemon thread ──────────────────────────────────
    # LS3 fix: pass session_id so the turn can inherit prior conversation history.
    orchestrator_thread = threading.Thread(
        target=run_live_streaming_session_ws,
        args=(on_payload, stop_event, session_id),
        daemon=True,
        name="ws-orchestrator",
    )
    orchestrator_thread.start()

    # ── Notify client ─────────────────────────────────────────────────────────
    session_note = f"Session: {session_id}" if session_id else "New session started."
    await websocket.send_json({
        "type":    "status",
        "status":  "connected",
        "message": f"Live stream active. Speak into the microphone. {session_note}",
    })

    # ── Keep-alive loop ───────────────────────────────────────────────────────
    try:
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                if msg.strip().lower() in ("stop", "disconnect", "end"):
                    break
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

    finally:
        stop_event.set()
        orchestrator_thread.join(timeout=8)
        try:
            await websocket.close()
        except Exception:
            pass
