"""
schemas/emotion.py
Conversation history fields removed from response schema.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The raw text to analyze for emotion.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID. Omit on first request.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I feel really happy today!",
                "session_id": "sess-a1b2c3d4"
            }
        }


class MultimodalStartRequest(BaseModel):
    pass


class MultimodalStopRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="The session_id returned by /analyze/multimodal/start.",
    )


# ── NESTED SUB-MODELS ─────────────────────────────────────────────────────────

class UserInput(BaseModel):
    text: str = Field(description="Submitted text.")
    timestamp: str = Field(description="ISO-8601 UTC timestamp.")


class EmotionAnalysis(BaseModel):
    dominant_emotion: str = Field(description="Top emotion after fusion.")
    confidence: float = Field(ge=0.0, le=1.0)
    emotion_probabilities: Dict[str, float] = Field(
        description="Probability distribution across 15 emotion classes."
    )


class ToneAnalysis(BaseModel):
    tone: str = Field(description="e.g. 'cheerful', 'somber', 'measured'.")
    confidence: float = Field(ge=0.0, le=1.0)


# ── UNIFIED RESPONSE MODEL ────────────────────────────────────────────────────

class UnifiedEmotionResponse(BaseModel):
    """
    V2 payload. Conversation history removed.
    Context window is managed by the LLM side.
    """
    session_id: str
    user_input: UserInput
    emotion_analysis: EmotionAnalysis
    tone_analysis: ToneAnalysis

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "session_id": "sess-a1b2c3d4",
                "user_input": {
                    "text": "I feel really happy today!",
                    "timestamp": "2026-05-10T00:00:00Z"
                },
                "emotion_analysis": {
                    "dominant_emotion": "joy",
                    "confidence": 0.87,
                    "emotion_probabilities": {
                        "happy": 0.87, "sad": 0.01, "angry": 0.01,
                        "surprised": 0.01, "neutral": 0.01, "fear": 0.01,
                        "empathetic": 0.01, "concerned": 0.01, "disgust": 0.01,
                        "shame": 0.01, "guilt": 0.01, "anxiety": 0.01,
                        "frustration": 0.01, "joy": 0.01, "calm": 0.01
                    }
                },
                "tone_analysis": {
                    "tone": "cheerful",
                    "confidence": 0.80
                }
            }
        }


VoiceAnalysisResponse = UnifiedEmotionResponse


# ── MULTIMODAL SESSION MODELS ─────────────────────────────────────────────────

class MultimodalSessionStarted(BaseModel):
    session_id: str
    status: str = Field(default="recording")
    max_duration_seconds: int = Field(default=30)
    message: str = Field(
        default="Recording started. Call POST /analyze/multimodal/stop when done."
    )


# ── ERROR MODEL ───────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = Field(default=None)