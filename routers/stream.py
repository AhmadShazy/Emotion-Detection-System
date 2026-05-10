"""
routers/stream.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter
from src.core.config import TEXT_ONLY_MODE

router = APIRouter()

if TEXT_ONLY_MODE:
    from fastapi.websockets import WebSocket

    @router.websocket("/ws/stream")
    async def stream_disabled(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({
            "type":    "error",
            "code":    "DISABLED",
            "message": "Live stream is disabled in text-only deployment mode."
        })
        await websocket.close(code=1011)

else:
    import asyncio
    import threading
    from fastapi import Query
    from fastapi.websockets import WebSocket, WebSocketDisconnect
    from typing import Optional
    from src.streaming.live_orchestrator import run_live_streaming_session_ws

    def _check_server_audio() -> tuple:
        try:
            import sounddevice as sd
            sd.check_input_settings(samplerate=16000, channels=1, dtype="float32")
            return True, ""
        except Exception as exc:
            return False, f"Server microphone unavailable: {exc}."

    @router.websocket("/ws/stream")
    async def live_stream(
        websocket: WebSocket,
        session_id: Optional[str] = Query(default=None),
    ):
        mic_ok, mic_error = _check_server_audio()
        if not mic_ok:
            await websocket.accept()
            await websocket.send_json({
                "type":    "error",
                "code":    "MIC_UNAVAILABLE",
                "message": mic_error,
            })
            await websocket.close(code=1011)
            return

        await websocket.accept()

        stop_event = threading.Event()
        loop       = asyncio.get_event_loop()

        def on_payload(payload: dict):
            if stop_event.is_set():
                return
            try:
                future = asyncio.run_coroutine_threadsafe(
                    websocket.send_json(payload), loop,
                )
                future.result(timeout=5)
            except Exception:
                stop_event.set()

        orchestrator_thread = threading.Thread(
            target=run_live_streaming_session_ws,
            args=(on_payload, stop_event, session_id),
            daemon=True,
            name="ws-orchestrator",
        )
        orchestrator_thread.start()

        await websocket.send_json({
            "type":    "status",
            "status":  "connected",
            "message": "Live stream active. Speak into the microphone.",
        })

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