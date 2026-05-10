"""
routers/text.py

POST /analyze/text
Chain:
    TextAnalysisRequest.text
        → process_text_emotion()
        → process_and_print_unified_json()
        → forward JSON to LLM endpoint (fire and forget)
        → return UnifiedEmotionResponse
"""

import sys
import os
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx
from fastapi import APIRouter, HTTPException

from schemas.emotion import TextAnalysisRequest, UnifiedEmotionResponse
from src.interactive_modes import process_text_emotion
from src.streaming.unified_pipeline import process_and_print_unified_json
from src.core.config import LLM_ENDPOINT_URL

router = APIRouter()


async def _forward_to_llm(payload: dict):
    """
    Fires the emotion payload to the LLM endpoint.
    Fire and forget — errors are logged but never crash the main request.
    Timeout: 5 seconds maximum.
    """
    if not LLM_ENDPOINT_URL or LLM_ENDPOINT_URL == "http://placeholder.url/endpoint":
        print("[TextRouter] ⚠️  LLM_ENDPOINT_URL not set — skipping forward.")
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                LLM_ENDPOINT_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            print(f"[TextRouter] ✅ Forwarded to LLM — status: {response.status_code}")
    except httpx.TimeoutException:
        print("[TextRouter] ⚠️  LLM forward timed out after 5s — continuing.")
    except Exception as e:
        print(f"[TextRouter] ⚠️  LLM forward failed: {e} — continuing.")


@router.post(
    "/text",
    response_model=UnifiedEmotionResponse,
    summary="Text Emotion Analysis",
)
async def analyze_text(request: TextAnalysisRequest):
    try:
        def _run():
            text_state = process_text_emotion(request.text)
            return process_and_print_unified_json(
                text_state=text_state,
                voice_state=None,
                face_state=None,
                raw_text=request.text,
                voice_emo_raw="neutral",
                face_emo_raw="neutral",
                session_id=request.session_id,
            )

        payload = await asyncio.to_thread(_run)

        # Forward to LLM — fire and forget, never blocks user response
        asyncio.create_task(_forward_to_llm(payload))

        return payload

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Text analysis pipeline error: {str(exc)}",
        ) from exc