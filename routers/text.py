"""
routers/text.py

POST /analyze/text
Accepts raw text → runs the full emotion pipeline → returns UnifiedEmotionResponse.

Chain:
    TextAnalysisRequest.text
        → process_text_emotion()           (interactive_modes.py)
        → process_and_print_unified_json() (unified_pipeline.py)
        → UnifiedEmotionResponse
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
from fastapi import APIRouter, HTTPException

from schemas.emotion import TextAnalysisRequest, UnifiedEmotionResponse
from src.interactive_modes import process_text_emotion
from src.streaming.unified_pipeline import process_and_print_unified_json

router = APIRouter()


@router.post(
    "/text",
    response_model=UnifiedEmotionResponse,
    summary="Text Emotion Analysis",
    description=(
        "Submit a piece of text and receive the full V2 emotion payload. "
        "Internally runs RoBERTa → Fusion Engine → LLM Adapter."
    ),
)
async def analyze_text(request: TextAnalysisRequest):
    """
    Runs text emotion analysis and returns the unified V2 JSON payload.
    Heavy ML inference is offloaded to a thread so the event loop stays free.
    """
    try:
        # Run blocking ML inference in a thread pool (keeps event loop unblocked)
        def _run():
            text_state = process_text_emotion(request.text)
            return process_and_print_unified_json(
                text_state=text_state,
                voice_state=None,
                face_state=None,
                raw_text=request.text,
                voice_emo_raw="neutral",
                face_emo_raw="neutral",
            )

        payload = await asyncio.to_thread(_run)
        return payload

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis pipeline error: {str(exc)}",
        ) from exc
