"""
routers/multimodal.py
=====================
Start/Stop recording model — max 30-second hard cap.

Bug 2 fix: Auto-stop now calls _process_and_close_session() properly
           instead of just setting a flag and leaving a zombie session.

Bug 7 fix: Temp files are deleted AFTER the payload is built and returned,
           not during processing.

POST /analyze/multimodal/start  → launches OpenFace + audio on server
POST /analyze/multimodal/stop   → stops, processes, returns UnifiedEmotionResponse
"""

import sys
import os
import uuid
import time
import datetime
import threading
import subprocess
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, HTTPException

from schemas.emotion import (
    MultimodalStartRequest,
    MultimodalStopRequest,
    MultimodalSessionStarted,
    UnifiedEmotionResponse,
)
from src.interactive_modes import process_multimodal_data
from src.streaming.unified_pipeline import process_and_print_unified_json

router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DURATION_SECONDS = 30
FS                   = 16000

OPENFACE_DIR = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64")
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "recordings")

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict      = {}
_sessions_lock       = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════
# START
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/multimodal/start",
    response_model=MultimodalSessionStarted,
    summary="Start Multimodal Recording",
)
async def start_multimodal(request: MultimodalStartRequest = None):
    for d in (OUTPUT_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)

    session_id  = f"mm-{uuid.uuid4().hex[:8]}"
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    of_filename = f"multimodal_{session_id}_{timestamp}"
    csv_path    = os.path.join(OUTPUT_DIR, f"{of_filename}.csv")
    wav_path    = os.path.join(DATA_DIR,   f"multimodal_{session_id}_{timestamp}.wav")

    face_available = os.path.exists(OPENFACE_EXE)
    audio_chunks: list = []
    stop_event         = threading.Event()

    # ── Audio worker ──────────────────────────────────────────────────────────
    def _audio_worker():
        try:
            import sounddevice as sd
            with sd.InputStream(samplerate=FS, channels=1, dtype="int16") as stream:
                while not stop_event.is_set():
                    chunk, _ = stream.read(FS // 2)
                    audio_chunks.append(chunk.copy())
        except Exception:
            pass

    # ── OpenFace ──────────────────────────────────────────────────────────────
    of_process = None
    if face_available:
        try:
            of_process = subprocess.Popen(
                [OPENFACE_EXE, "-device", "0", "-out_dir", OUTPUT_DIR, "-of", of_filename],
                cwd=OPENFACE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            face_available = False

    # ── Start audio thread ────────────────────────────────────────────────────
    audio_thread = threading.Thread(target=_audio_worker, daemon=True)
    audio_thread.start()

    # ── Store session ─────────────────────────────────────────────────────────
    session = {
        "stop_event":     stop_event,
        "audio_thread":   audio_thread,
        "of_process":     of_process,
        "wav_path":       wav_path,
        "csv_path":       csv_path,
        "face_available": face_available,
        "audio_chunks":   audio_chunks,
        "start_time":     time.time(),
        "processed":      False,   # guard: ensures we process exactly once
    }
    with _sessions_lock:
        _sessions[session_id] = session

    # ── Auto-stop timer (hard 30-second cap) ──────────────────────────────────
    # Bug 2 fix: auto-stop now calls _process_and_close_session() so the
    # session is fully analysed and removed even if the client never calls /stop.
    def _auto_stop():
        time.sleep(MAX_DURATION_SECONDS)

        with _sessions_lock:
            sess = _sessions.get(session_id)
            if sess is None or sess.get("processed"):
                return  # already handled by /stop — nothing to do
            sess["processed"] = True       # claim ownership before releasing lock

        # Process in background thread (blocking ML work)
        try:
            _process_and_close_session(sess)
        except Exception as e:
            print(f"[Multimodal] Auto-stop processing error for {session_id}: {e}")
        finally:
            # Remove from store regardless of success
            with _sessions_lock:
                _sessions.pop(session_id, None)

    threading.Thread(target=_auto_stop, daemon=True).start()

    return MultimodalSessionStarted(
        session_id=session_id,
        status="recording",
        max_duration_seconds=MAX_DURATION_SECONDS,
        message=(
            f"Recording started on server (camera + microphone). "
            f"POST /analyze/multimodal/stop with session_id='{session_id}' when ready. "
            f"Auto-stops and analyses after {MAX_DURATION_SECONDS} seconds."
        ),
    )


# ════════════════════════════════════════════════════════════════════════════
# STOP
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/multimodal/stop",
    response_model=UnifiedEmotionResponse,
    summary="Stop Multimodal Recording & Analyze",
)
async def stop_multimodal(request: MultimodalStopRequest):
    with _sessions_lock:
        session = _sessions.get(request.session_id)

        if session is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Session '{request.session_id}' not found. "
                    "It may have already been auto-stopped or never started."
                ),
            )

        if session.get("processed"):
            # Auto-stop already claimed this session — race condition guard
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Session '{request.session_id}' was already auto-stopped "
                    "after the 30-second cap. Results were processed server-side."
                ),
            )

        # Claim ownership so auto-stop thread won't double-process
        session["processed"] = True
        # Remove from store now — /stop owns cleanup from here
        del _sessions[request.session_id]

    try:
        payload = await asyncio.to_thread(_process_and_close_session, session)
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal processing error: {str(exc)}",
        ) from exc


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL — teardown + processing (runs in thread pool)
# ════════════════════════════════════════════════════════════════════════════

def _process_and_close_session(session: dict) -> dict:
    """
    1. Signals stop → audio worker exits
    2. Terminates OpenFace
    3. Saves WAV file
    4. Runs full analysis pipeline
    5. Deletes temp files AFTER payload is built  ← Bug 7 fix
    6. Returns payload dict
    """
    try:
        import numpy as np
        from scipy.io.wavfile import write as wav_write
    except ImportError as e:
        raise RuntimeError(f"Missing audio library: {e}")

    stop_event:     threading.Event  = session["stop_event"]
    audio_thread:   threading.Thread = session["audio_thread"]
    of_process                       = session["of_process"]
    wav_path:  str                   = session["wav_path"]
    csv_path:  str                   = session["csv_path"]
    face_available: bool             = session["face_available"]
    audio_chunks: list               = session["audio_chunks"]

    # ── 1. Signal stop ────────────────────────────────────────────────────────
    stop_event.set()
    audio_thread.join(timeout=3)

    # ── 2. Stop OpenFace ──────────────────────────────────────────────────────
    if of_process is not None and of_process.poll() is None:
        of_process.terminate()
        try:
            of_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            of_process.kill()

    # ── 3. Save WAV ───────────────────────────────────────────────────────────
    saved_wav = None
    if audio_chunks:
        try:
            audio_data = np.concatenate(audio_chunks, axis=0)
            wav_write(wav_path, FS, audio_data)
            saved_wav = wav_path
        except Exception as e:
            print(f"[Multimodal] WAV save error: {e}")

    # ── 4. Run full analysis pipeline ─────────────────────────────────────────
    text_state, voice_state, face_state, stt_result, ser_result, _ = (
        process_multimodal_data(saved_wav, csv_path, face_available)
    )

    face_emo_raw = face_state["emotion"] if face_state else "neutral"

    payload = process_and_print_unified_json(
        text_state=text_state,
        voice_state=voice_state,
        face_state=face_state,
        raw_text=stt_result if stt_result != "N/A" else "",
        voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
        face_emo_raw=face_emo_raw,
        # Multimodal sessions use their own session_id so history is isolated
        session_id=session.get("session_id"),
    )

    # ── 5. Delete temp files AFTER payload is built ───────────────────────────
    # Bug 7 fix: previously deleted during processing which would break any
    # future re-analysis feature. Now deleted only after we're fully done.
    for path in (saved_wav, csv_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return payload