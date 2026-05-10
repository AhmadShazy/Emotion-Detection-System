"""
routers/multimodal.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, HTTPException
from src.core.config import TEXT_ONLY_MODE

router = APIRouter()

if TEXT_ONLY_MODE:
    @router.post("/multimodal/start")
    async def multimodal_start_disabled():
        raise HTTPException(
            status_code=503,
            detail="Multimodal analysis is disabled in text-only deployment mode."
        )

    @router.post("/multimodal/stop")
    async def multimodal_stop_disabled():
        raise HTTPException(
            status_code=503,
            detail="Multimodal analysis is disabled in text-only deployment mode."
        )
else:
    import uuid
    import time
    import datetime
    import threading
    import subprocess
    import asyncio

    from schemas.emotion import (
        MultimodalStartRequest, MultimodalStopRequest,
        MultimodalSessionStarted, UnifiedEmotionResponse,
    )
    from src.interactive_modes import process_multimodal_data
    from src.streaming.unified_pipeline import process_and_print_unified_json

    MAX_DURATION_SECONDS = 30
    FS                   = 16000
    OPENFACE_DIR = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64")
    OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
    DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "recordings")

    _sessions: dict    = {}
    _sessions_lock     = threading.Lock()

    @router.post("/multimodal/start", response_model=MultimodalSessionStarted)
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

        def _audio_worker():
            try:
                import sounddevice as sd
                with sd.InputStream(samplerate=FS, channels=1, dtype="int16") as stream:
                    while not stop_event.is_set():
                        chunk, _ = stream.read(FS // 2)
                        audio_chunks.append(chunk.copy())
            except Exception:
                pass

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

        audio_thread = threading.Thread(target=_audio_worker, daemon=True)
        audio_thread.start()

        session = {
            "session_id":      session_id,
            "processing_lock": threading.Lock(),
            "stop_event":      stop_event,
            "audio_thread":    audio_thread,
            "of_process":      of_process,
            "wav_path":        wav_path,
            "csv_path":        csv_path,
            "face_available":  face_available,
            "audio_chunks":    audio_chunks,
            "start_time":      time.time(),
            "processed":       False,
        }
        with _sessions_lock:
            _sessions[session_id] = session

        def _auto_stop():
            time.sleep(MAX_DURATION_SECONDS)
            with _sessions_lock:
                sess = _sessions.get(session_id)
                if sess is None:
                    return
            acquired = sess["processing_lock"].acquire(blocking=False)
            if not acquired:
                return
            try:
                sess["processed"] = True
                with _sessions_lock:
                    _sessions.pop(session_id, None)
                _process_and_close_session(sess)
            except Exception as e:
                print(f"[Multimodal] Auto-stop error: {e}")
            finally:
                sess["processing_lock"].release()

        threading.Thread(target=_auto_stop, daemon=True).start()

        return MultimodalSessionStarted(
            session_id=session_id,
            status="recording",
            max_duration_seconds=MAX_DURATION_SECONDS,
            message=f"Recording started. POST /analyze/multimodal/stop with session_id='{session_id}'.",
        )

    @router.post("/multimodal/stop", response_model=UnifiedEmotionResponse)
    async def stop_multimodal(request: MultimodalStopRequest):
        with _sessions_lock:
            session = _sessions.get(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

        acquired = session["processing_lock"].acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=409, detail="Session already being processed.")

        try:
            session["processed"] = True
            with _sessions_lock:
                _sessions.pop(request.session_id, None)
            payload = await asyncio.to_thread(_process_and_close_session, session)
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            session["processing_lock"].release()

    def _process_and_close_session(session: dict) -> dict:
        import numpy as np
        from scipy.io.wavfile import write as wav_write

        stop_event   = session["stop_event"]
        audio_thread = session["audio_thread"]
        of_process   = session["of_process"]
        wav_path     = session["wav_path"]
        csv_path     = session["csv_path"]
        face_available = session["face_available"]
        audio_chunks   = session["audio_chunks"]

        stop_event.set()
        audio_thread.join(timeout=3)

        if of_process is not None and of_process.poll() is None:
            of_process.terminate()
            try:
                of_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                of_process.kill()

        saved_wav = None
        if audio_chunks:
            try:
                audio_data = np.concatenate(audio_chunks, axis=0)
                wav_write(wav_path, FS, audio_data)
                saved_wav = wav_path
            except Exception as e:
                print(f"[Multimodal] WAV save error: {e}")

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
            session_id=session.get("session_id"),
        )

        for path in (saved_wav, csv_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        return payload