"""
routers/stream.py

WebSocket endpoint: ws://localhost:8000/ws/stream

Flow:
    1. Client connects (browser opens WebSocket)
    2. Server accepts connection + starts run_live_streaming_session_ws()
       in a background thread.
    3. Each time a speech turn completes, the pipeline calls on_payload(dict)
       which thread-safely sends the JSON to the connected client.
    4. The async loop waits for any message from the client (used as a
       "keep-alive" or "ping") — or for a WebSocketDisconnect.
    5. On disconnect (or any error), stop_event is set → workers shut down
       gracefully in the background thread.
"""

import sys
import os
import asyncio
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect

from src.streaming.live_orchestrator import run_live_streaming_session_ws

router = APIRouter()


@router.websocket("/ws/stream")
async def live_stream(websocket: WebSocket):
    """
    WebSocket handler for Option 4 — Live Multimodal Streaming.

    The orchestrator runs in a daemon thread. Each completed speech turn
    produces a payload that is pushed to the client via send_json().
    Disconnecting the client (closing the browser tab or calling ws.close())
    cleanly shuts down all audio/face/STT workers.
    """
    await websocket.accept()

    # ── Shared state between async loop and orchestrator thread ──────────────
    stop_event = threading.Event()
    loop       = asyncio.get_event_loop()

    # ── Thread-safe payload sender ────────────────────────────────────────────
    # The orchestrator runs in a non-async thread. It calls on_payload(dict)
    # synchronously. We bridge back to the async event loop with
    # run_coroutine_threadsafe so WebSocket.send_json() is called correctly.
    def on_payload(payload: dict):
        """Called from the background thread after each turn — sends JSON to client."""
        if stop_event.is_set():
            return  # client already disconnected — skip silently
        try:
            future = asyncio.run_coroutine_threadsafe(
                websocket.send_json(payload),
                loop,
            )
            future.result(timeout=5)  # wait up to 5 s for the send to complete
        except Exception:
            # WebSocket may have closed between the check and the send
            stop_event.set()

    # ── Launch orchestrator in daemon thread ──────────────────────────────────
    orchestrator_thread = threading.Thread(
        target=run_live_streaming_session_ws,
        args=(on_payload, stop_event),
        daemon=True,
        name="ws-orchestrator",
    )
    orchestrator_thread.start()

    # ── Send a ready signal to the client ─────────────────────────────────────
    await websocket.send_json({
        "type": "status",
        "status": "connected",
        "message": "Live stream active. Speak into the microphone.",
    })

    # ── Keep-alive loop ───────────────────────────────────────────────────────
    # We poll for client messages in short windows so we can detect disconnect
    # quickly without blocking indefinitely.
    try:
        while not stop_event.is_set():
            try:
                # Wait up to 1 s for any client message (ping / control frame)
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Future: handle "stop", "ping" control messages here
                if msg.strip().lower() in ("stop", "disconnect", "end"):
                    break
            except asyncio.TimeoutError:
                continue  # no message yet — keep waiting
            except WebSocketDisconnect:
                break

    finally:
        # ── Graceful shutdown ─────────────────────────────────────────────────
        stop_event.set()
        orchestrator_thread.join(timeout=8)  # give workers time to stop cleanly
        # Best-effort close in case connection is still open
        try:
            await websocket.close()
        except Exception:
            pass
