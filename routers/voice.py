"""
routers/voice.py
"""

import os
import sys
import uuid
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from schemas.emotion import UnifiedEmotionResponse
from src.core.config import TEXT_ONLY_MODE

router = APIRouter()

if TEXT_ONLY_MODE:
    @router.post("/voice")
    async def voice_disabled():
        raise HTTPException(
            status_code=503,
            detail="Voice analysis is disabled in text-only deployment mode."
        )
else:
    from src.interactive_modes import process_voice_pipeline, _check_audio_has_speech, _is_hallucination
    from src.video.ingest import ensure_16k_mono
    from src.streaming.unified_pipeline import process_and_print_unified_json

    _UPLOAD_DIR   = os.path.join(PROJECT_ROOT, "data", "recordings")
    _ALLOWED_TYPES = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/vnd.wave", "application/octet-stream",
    }
    _MIN_WAV_BYTES = 16000

    @router.post(
        "/voice",
        response_model=UnifiedEmotionResponse,
        summary="Voice Emotion Analysis",
    )
    async def analyze_voice(
        file: UploadFile = File(...),
        session_id: Optional[str] = Form(default=None),
    ):
        content_type = (file.content_type or "").lower()
        filename     = (file.filename    or "").lower()

        if content_type not in _ALLOWED_TYPES and not filename.endswith(".wav"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Upload a .wav file.",
            )

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(audio_bytes) < _MIN_WAV_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Recording too short. Please speak for at least 1 second.",
            )

        os.makedirs(_UPLOAD_DIR, exist_ok=True)
        wav_path = os.path.join(_UPLOAD_DIR, f"api_voice_{uuid.uuid4().hex}.wav")

        try:
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

            # ── Normalise to 16 kHz mono ────────────────────────────────────
            # This route accepts any .wav a user can produce, and SpeechBrain
            # expects 16 kHz. Nothing downstream resamples, and handing
            # wav2vec2 the wrong rate does not raise — it returns a CONFIDENT
            # WRONG label, which is the worst possible failure. The video and
            # live paths already guarantee 16 kHz (ffmpeg -ar / the browser's
            # AudioContext); this one did not.
            wav_path = await asyncio.to_thread(ensure_16k_mono, wav_path)

            has_speech = await asyncio.to_thread(_check_audio_has_speech, wav_path)
            if not has_speech:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No speech was detected in that recording. Check your "
                        "microphone is working and try speaking a little louder."
                    ),
                )

            def _run():
                text_state, voice_state, stt_result, ser_result = (
                    process_voice_pipeline(wav_path)
                )
                effective_text = (
                    stt_result
                    if stt_result and stt_result != "N/A"
                       and not _is_hallucination(stt_result)
                    else ""
                )
                return process_and_print_unified_json(
                    text_state=text_state,
                    voice_state=voice_state,
                    face_state=None,
                    raw_text=effective_text,
                    voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
                    face_emo_raw="neutral",
                    session_id=session_id,
                )

            return await asyncio.to_thread(_run)

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Voice analysis pipeline error: {str(exc)}",
            ) from exc
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass