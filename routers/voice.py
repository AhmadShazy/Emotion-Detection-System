"""
routers/voice.py

POST /analyze/voice
Accepts a .wav upload + optional session_id → SER + Whisper STT + Text Emotion
→ UnifiedEmotionResponse.

Chain:
    UploadFile (.wav)
        → save to temp path (uuid-named)
        → process_voice_pipeline()         (interactive_modes.py)
        → process_and_print_unified_json() (unified_pipeline.py)
        → delete temp file (finally block)
        → UnifiedEmotionResponse
"""

import sys
import os
import uuid
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from schemas.emotion import UnifiedEmotionResponse
from src.interactive_modes import process_voice_pipeline
from src.streaming.unified_pipeline import process_and_print_unified_json

router = APIRouter()

_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")

_ALLOWED_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


@router.post(
    "/voice",
    response_model=UnifiedEmotionResponse,
    summary="Voice Emotion Analysis",
    description=(
        "Upload a .wav audio file. The server runs Speech Emotion Recognition "
        "(Wav2Vec2 / SER), Speech-to-Text (Whisper), and Text Emotion Analysis "
        "(RoBERTa), then returns the fused V2 emotion payload. "
        "Pass the same `session_id` form field across requests to maintain "
        "conversation history. Omit it to auto-create a new session."
    ),
)
async def analyze_voice(
    file: UploadFile = File(
        ...,
        description="A WAV audio file to analyze.",
    ),
    session_id: Optional[str] = Form(
        default=None,
        description=(
            "Optional session ID returned by a previous request. "
            "Enables conversation history and emotion trend tracking across turns."
        ),
    ),
):
    """
    Saves the uploaded WAV temporarily, processes it through the full voice
    pipeline, then deletes the temp file — even if processing fails.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    filename     = (file.filename or "").lower()

    if content_type not in _ALLOWED_TYPES and not filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a .wav audio file."
            ),
        )

    # ── Prepare temp path ─────────────────────────────────────────────────────
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    wav_path = os.path.join(_UPLOAD_DIR, f"api_voice_{uuid.uuid4().hex}.wav")

    try:
        # ── Save bytes to disk ────────────────────────────────────────────────
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # ── Run pipeline in thread ────────────────────────────────────────────
        def _run():
            text_state, voice_state, stt_result, ser_result = process_voice_pipeline(wav_path)
            return process_and_print_unified_json(
                text_state=text_state,
                voice_state=voice_state,
                face_state=None,
                raw_text=stt_result if stt_result != "N/A" else "",
                voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
                face_emo_raw="neutral",
                session_id=session_id,   # None = auto-create new session
            )

        payload = await asyncio.to_thread(_run)
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Voice analysis pipeline error: {str(exc)}",
        ) from exc
    finally:
        # ── Always clean up temp file ─────────────────────────────────────────
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass