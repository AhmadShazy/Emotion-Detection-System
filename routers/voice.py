"""
routers/voice.py
=================
POST /analyze/voice
Fixes applied:
  - Minimum file size check (rejects < 0.5s recordings before pipeline)
  - Audio energy pre-check (silence detection before Whisper)
  - Graceful N/A handling when STT fails
  - Clear 4xx errors shown as toast in frontend
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
from src.interactive_modes import (
    process_voice_pipeline,
    _check_audio_has_speech,
    _is_hallucination,
)
from src.streaming.unified_pipeline import process_and_print_unified_json

router = APIRouter()

_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")

_ALLOWED_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/vnd.wave", "application/octet-stream",
}

# 16000 bytes ≈ 0.5 s of 16kHz 16-bit mono (absolute floor)
_MIN_WAV_BYTES = 16000


@router.post(
    "/voice",
    response_model=UnifiedEmotionResponse,
    summary="Voice Emotion Analysis",
    description=(
        "Upload a .wav audio file. The server runs Speech Emotion Recognition "
        "(Wav2Vec2), Speech-to-Text (Whisper), and Text Emotion Analysis "
        "(RoBERTa), then returns the fused V2 emotion payload."
    ),
)
async def analyze_voice(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    # ── Validate file type ────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    filename     = (file.filename    or "").lower()

    if content_type not in _ALLOWED_TYPES and not filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a .wav file.",
        )

    # ── Read bytes ────────────────────────────────────────────────────────────
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Minimum size check (fast-fail before disk write) ──────────────────────
    if len(audio_bytes) < _MIN_WAV_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recording too short ({len(audio_bytes)} bytes). "
                "Please speak for at least 1 second."
            ),
        )

    # ── Save to temp path ─────────────────────────────────────────────────────
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    wav_path = os.path.join(_UPLOAD_DIR, f"api_voice_{uuid.uuid4().hex}.wav")

    try:
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # ── Silence check — prevents Whisper hallucination + SER noise ────────
        has_speech = await asyncio.to_thread(_check_audio_has_speech, wav_path)
        if not has_speech:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No speech detected in the recording. "
                    "Ensure your microphone is working and speak clearly."
                ),
            )

        # ── Run pipeline in thread ────────────────────────────────────────────
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