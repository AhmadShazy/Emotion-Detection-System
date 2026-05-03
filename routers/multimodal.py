"""
routers/multimodal.py

Start/Stop recording model — max 30-second hard cap.

POST /analyze/multimodal/start
    → Launches OpenFace subprocess + audio recording thread on the SERVER machine.
    → Returns { session_id } immediately so the frontend can show a live timer.

POST /analyze/multimodal/stop
    → Accepts { session_id }
    → Signals stop, saves WAV, runs process_multimodal_data() + unified pipeline.
    → Returns UnifiedEmotionResponse.

Server auto-stops any session after MAX_DURATION_SECONDS (30 s) even if the
client never calls /stop, preventing zombie recordings.
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

# ── Constants ────────────────────────────────────────────────────────────────
MAX_DURATION_SECONDS = 30
FS = 16000  # audio sample rate

OPENFACE_DIR = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64")
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "recordings")

# ── In-memory session store ──────────────────────────────────────────────────
# { session_id: { stop_event, audio_thread, of_process, wav_path, csv_path,
#                 face_available, audio_chunks, start_time } }
_sessions: dict = {}
_sessions_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════
# START ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/multimodal/start",
    response_model=MultimodalSessionStarted,
    summary="Start Multimodal Recording",
    description=(
        "Starts simultaneous OpenFace (face) + microphone (audio) recording "
        "on the **server machine**. Returns a `session_id` immediately. "
        "Call `/analyze/multimodal/stop` with that ID when done. "
        "Server enforces a 30-second hard cap and auto-stops if not called."
    ),
)
async def start_multimodal(request: MultimodalStartRequest = None):
    """
    Launches server-side camera + audio recording.
    Responds instantly so the frontend can start its elapsed-time timer.
    """
    # Ensure output directories exist
    for d in (OUTPUT_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)

    session_id = f"mm-{uuid.uuid4().hex[:8]}"
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    of_filename = f"multimodal_{session_id}_{timestamp}"
    csv_path    = os.path.join(OUTPUT_DIR, f"{of_filename}.csv")
    wav_path    = os.path.join(DATA_DIR,   f"multimodal_{session_id}_{timestamp}.wav")

    face_available = os.path.exists(OPENFACE_EXE)

    # ── Audio setup ──────────────────────────────────────────────────────────
    audio_chunks: list = []
    stop_event = threading.Event()

    def _audio_worker():
        try:
            import sounddevice as sd
            with sd.InputStream(samplerate=FS, channels=1, dtype="int16") as stream:
                while not stop_event.is_set():
                    chunk, _ = stream.read(FS // 2)
                    audio_chunks.append(chunk.copy())
        except Exception:
            pass  # audio unavailable — session still returns face-only results

    # ── Launch OpenFace (non-blocking) ────────────────────────────────────────
    of_process = None
    if face_available:
        of_cmd = [
            OPENFACE_EXE,
            "-device", "0",
            "-out_dir", OUTPUT_DIR,
            "-of", of_filename,
        ]
        try:
            of_process = subprocess.Popen(
                of_cmd,
                cwd=OPENFACE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            face_available = False

    # ── Start audio thread ───────────────────────────────────────────────────
    audio_thread = threading.Thread(target=_audio_worker, daemon=True)
    audio_thread.start()

    # ── Store session ────────────────────────────────────────────────────────
    session = {
        "stop_event":      stop_event,
        "audio_thread":    audio_thread,
        "of_process":      of_process,
        "wav_path":        wav_path,
        "csv_path":        csv_path,
        "face_available":  face_available,
        "audio_chunks":    audio_chunks,
        "start_time":      time.time(),
    }
    with _sessions_lock:
        _sessions[session_id] = session

    # ── Auto-stop timer (30-second hard cap) ─────────────────────────────────
    def _auto_stop():
        time.sleep(MAX_DURATION_SECONDS)
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["stop_event"].set()
                # Mark so /stop knows it was auto-triggered
                _sessions[session_id]["auto_stopped"] = True

    threading.Thread(target=_auto_stop, daemon=True).start()

    return MultimodalSessionStarted(
        session_id=session_id,
        status="recording",
        max_duration_seconds=MAX_DURATION_SECONDS,
        message=(
            "Recording started on server (camera + microphone). "
            "POST /analyze/multimodal/stop with this session_id when ready. "
            f"Auto-stops after {MAX_DURATION_SECONDS} seconds."
        ),
    )


# ════════════════════════════════════════════════════════════════════════════
# STOP ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@router.post(
    "/multimodal/stop",
    response_model=UnifiedEmotionResponse,
    summary="Stop Multimodal Recording & Analyze",
    description=(
        "Stops the active recording identified by `session_id`, processes "
        "the captured audio + face data through the full pipeline, and "
        "returns the unified emotion analysis payload."
    ),
)
async def stop_multimodal(request: MultimodalStopRequest):
    """
    Signals the recording to stop, saves the WAV file, runs post-processing
    (SER → Whisper → RoBERTa → Face → Fusion), and returns the result.
    Heavy processing is offloaded to a thread pool.
    """
    with _sessions_lock:
        session = _sessions.pop(request.session_id, None)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{request.session_id}' not found. "
                "It may have already been stopped or never started."
            ),
        )

    # Run all blocking work in a thread (Whisper / SER / OpenFace parsing)
    try:
        payload = await asyncio.to_thread(_process_and_close_session, session)
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal processing error: {str(exc)}",
        ) from exc


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL — session teardown + processing (runs in thread pool)
# ════════════════════════════════════════════════════════════════════════════

def _process_and_close_session(session: dict) -> dict:
    """
    Called inside asyncio.to_thread — can safely block here.

    1. Signals stop_event → audio worker exits.
    2. Terminates OpenFace.
    3. Saves WAV file.
    4. Runs process_multimodal_data() → process_and_print_unified_json().
    5. Returns the payload dict.
    """
    try:
        import numpy as np
        from scipy.io.wavfile import write as wav_write
    except ImportError as e:
        raise RuntimeError(f"Missing audio library: {e}")

    stop_event:   threading.Event = session["stop_event"]
    audio_thread: threading.Thread = session["audio_thread"]
    of_process    = session["of_process"]
    wav_path:  str = session["wav_path"]
    csv_path:  str = session["csv_path"]
    face_available: bool = session["face_available"]
    audio_chunks: list = session["audio_chunks"]

    # ── 1. Signal stop ───────────────────────────────────────────────────────
    stop_event.set()
    audio_thread.join(timeout=3)

    # ── 2. Stop OpenFace ─────────────────────────────────────────────────────
    if of_process is not None and of_process.poll() is None:
        of_process.terminate()
        try:
            of_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            of_process.kill()

    # ── 3. Save WAV ──────────────────────────────────────────────────────────
    saved_wav = None
    if audio_chunks:
        try:
            audio_data = np.concatenate(audio_chunks, axis=0)
            wav_write(wav_path, FS, audio_data)
            saved_wav = wav_path
        except Exception:
            pass  # no audio → face-only analysis

    # ── 4. Post-process ──────────────────────────────────────────────────────
    text_state, voice_state, face_state, stt_result, ser_result, _ = process_multimodal_data(
        saved_wav, csv_path, face_available
    )

    face_emo_raw = face_state["emotion"] if face_state else "neutral"

    payload = process_and_print_unified_json(
        text_state=text_state,
        voice_state=voice_state,
        face_state=face_state,
        raw_text=stt_result if stt_result != "N/A" else "",
        voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
        face_emo_raw=face_emo_raw,
    )

    # ── 5. Cleanup temp files ────────────────────────────────────────────────
    for path in [saved_wav, csv_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return payload
