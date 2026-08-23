"""
routers/stream.py
=================
WebSocket endpoint for the live call: ws://HOST/ws/stream

The browser captures its own microphone and camera and streams them here. The
server owns no capture device, which is what allows more than one person to use
this at once — the previous design opened the SERVER's microphone and webcam,
so it could only ever serve whoever was sitting at the machine.

Wire protocol
-------------
BINARY frames carry media, with a 1-byte type prefix:
    0x01 + payload   PCM audio, signed 16-bit LE, 16 kHz, mono
    0x02 + payload   one JPEG still

TEXT frames carry JSON control, both directions:
    -> {"type": "stop"}                    caller hangs up
    <- {"type": "status",  "code": ...}    connection / turn state
    <- {"type": "error",   "code": ...}    fatal, socket closes after
    <- the emotion payload itself (identified by its "session_id" key)

Concurrency
-----------
The connection's own coroutine is the ONLY writer to the socket. Analysis is
handed to a small shared executor and awaited, so the payload is sent by the
same coroutine that requested it. Nothing writes from a thread, which removes
the interleaving the old `run_coroutine_threadsafe` bridge risked.
"""

import os
import sys
import asyncio
import concurrent.futures

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
            "message": "Live streaming is disabled in this text-only deployment.",
        })
        await websocket.close(code=1011)

else:
    from typing import Optional
    from fastapi import Query
    from fastapi.websockets import WebSocket, WebSocketDisconnect

    from src.streaming.call_session import CallSession, analyze_turn

    # ── Message types ────────────────────────────────────────────────────────
    MSG_AUDIO = 0x01
    MSG_VIDEO = 0x02

    # ── Shared inference executor ────────────────────────────────────────────
    # Deliberately small, and deliberately NOT asyncio.to_thread (whose default
    # executor is min(32, cpu+4) — far too wide here). Whisper and SpeechBrain
    # saturate cores individually; running many concurrently measured SLOWER
    # than running them one after another, so extra workers would buy negative
    # throughput while multiplying latency.
    INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "2"))
    _executor: concurrent.futures.ThreadPoolExecutor | None = None

    # Refuse calls the machine cannot actually serve. A 5s turn costs ~4-5s of
    # CPU, so the honest capacity here is small — telling someone the server is
    # busy beats accepting them and delivering minute-long latency.
    MAX_ACTIVE_CALLS = int(os.environ.get("MAX_ACTIVE_CALLS", "2"))
    _active_calls = 0
    _active_lock = asyncio.Lock()

    def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
        global _executor
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=INFERENCE_WORKERS,
                thread_name_prefix="inference",
            )
        return _executor

    # ── Endpoint ─────────────────────────────────────────────────────────────

    @router.websocket("/ws/stream")
    async def live_stream(
        websocket: WebSocket,
        session_id: Optional[str] = Query(default=None),
    ):
        global _active_calls

        async with _active_lock:
            at_capacity = _active_calls >= MAX_ACTIVE_CALLS
            if not at_capacity:
                _active_calls += 1

        if at_capacity:
            await websocket.accept()
            await websocket.send_json({
                "type":    "error",
                "code":    "AT_CAPACITY",
                "message": (
                    f"The server is already handling {MAX_ACTIVE_CALLS} calls. "
                    f"Please try again shortly."
                ),
            })
            await websocket.close(code=1013)
            return

        await websocket.accept()

        session = CallSession(session_id=session_id)
        loop = asyncio.get_running_loop()
        pending: asyncio.Task | None = None

        await websocket.send_json({
            "type":    "status",
            "code":    "CONNECTED",
            "message": "Connected. Start speaking — results arrive after each pause.",
            "config": {
                "sample_rate":   16000,
                "audio_format":  "pcm_s16le",
                "video_format":  "jpeg",
                "target_fps":    3,
            },
        })

        async def send_payload(task: asyncio.Task):
            """Delivers one finished analysis, or reports why it failed."""
            try:
                payload = task.result()
                session.turns_emitted += 1
                await websocket.send_json(payload)
            except Exception as exc:
                print(f"[Live] Turn analysis failed: {exc}")
                try:
                    await websocket.send_json({
                        "type":    "status",
                        "code":    "TURN_FAILED",
                        "message": "That turn could not be analysed.",
                    })
                except Exception:
                    pass
            finally:
                session.analysis_in_flight = False

        try:
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # ── Control ──────────────────────────────────────────────────
                if (text := message.get("text")) is not None:
                    if '"stop"' in text or text.strip().lower() in ("stop", "end"):
                        # Hanging up mid-sentence still counts as having said
                        # something. Flush HERE, while the socket is certainly
                        # alive — doing it in `finally` races the close and
                        # there would be nobody left to send the result to.
                        final_turn = session.detector.flush()
                        if final_turn is not None and not session.analysis_in_flight:
                            final_turn["video"] = session._take_video_for_turn(
                                final_turn["duration"]
                            )
                            await websocket.send_json({
                                "type":     "status",
                                "code":     "ANALYZING",
                                "message":  "Analysing your last turn...",
                                "duration": round(final_turn["duration"], 1),
                            })
                            try:
                                payload = await loop.run_in_executor(
                                    _get_executor(),
                                    analyze_turn,
                                    final_turn["audio"],
                                    final_turn["video"],
                                    session.session_id,
                                )
                                await websocket.send_json(payload)
                                session.turns_emitted += 1
                            except Exception as exc:
                                print(f"[Live] Final turn failed: {exc}")
                        break
                    continue

                data = message.get("bytes")
                if not data:
                    continue

                kind, payload = data[0], data[1:]

                # ── Video ────────────────────────────────────────────────────
                if kind == MSG_VIDEO:
                    session.push_video(payload)
                    continue

                if kind != MSG_AUDIO:
                    continue

                # ── Audio ────────────────────────────────────────────────────
                turn = session.push_audio(payload)

                # A turn that outgrew the buffer is abandoned rather than
                # spliced — a transcript stitched across a gap reads as a real
                # sentence while being wrong.
                if turn is None and session.check_overflow():
                    session.detector.abandon_turn()
                    session.turns_shed += 1
                    await websocket.send_json({
                        "type":    "status",
                        "code":    "TURN_TOO_LONG",
                        "message": "That went on too long to analyse — please pause between thoughts.",
                    })
                    continue

                if turn is None:
                    continue

                # One turn in flight at a time. The next one is shed with an
                # honest message rather than queued behind it.
                if session.analysis_in_flight:
                    session.turns_shed += 1
                    await websocket.send_json({
                        "type":    "status",
                        "code":    "BUSY",
                        "message": "Still working on your last turn — this one was skipped.",
                    })
                    continue

                session.analysis_in_flight = True
                await websocket.send_json({
                    "type":     "status",
                    "code":     "ANALYZING",
                    "message":  "Analysing...",
                    "duration": round(turn["duration"], 1),
                })

                pending = loop.create_task(
                    asyncio.wrap_future(
                        _get_executor().submit(
                            analyze_turn,
                            turn["audio"],
                            turn["video"],
                            session.session_id,
                        )
                    )
                )
                pending.add_done_callback(
                    lambda t: loop.create_task(send_payload(t))
                )

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[Live] Connection error: {exc}")
        finally:
            session.closed = True

            # An abrupt disconnect (tab closed, network dropped) leaves speech
            # in the detector. There is nobody to send a result to, so it is
            # simply discarded — but say so, because "turns_shed" in the log
            # should account for it rather than it vanishing silently.
            abandoned = session.detector.abandon_turn()
            if abandoned:
                session.turns_shed += 1

            # Let an in-flight turn finish briefly rather than orphaning the
            # executor thread mid-inference.
            if pending is not None and not pending.done():
                try:
                    await asyncio.wait_for(asyncio.shield(pending), timeout=2.0)
                except Exception:
                    pass

            print(f"[Live] Call ended: {session.stats()}")

            async with _active_lock:
                _active_calls = max(0, _active_calls - 1)

            try:
                await websocket.close()
            except Exception:
                pass
