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


class ConflictAnalysis(BaseModel):
    """
    Reports a mismatch between what the face shows and what the voice conveys.

    This is the signal that distinguishes someone who IS angry from someone who
    is HIDING that they are angry — the two call for very different replies.
    Always present; `detected` is false when the signals agree or when fewer
    than two of them were available.
    """
    detected: bool = Field(
        description="True when the face and voice point at different emotions."
    )
    type: str = Field(
        description=(
            "Which mismatch was found: 'masked_anger' (happy face, angry voice), "
            "'suppressed_frustration' (neutral face, angry voice), "
            "'masked_sadness' (happy face, sad voice), "
            "'internal_sadness' (sad face, neutral voice), or 'none'."
        ),
        examples=["masked_anger"],
    )
    details: str = Field(
        description="Human-readable explanation. Empty string when none was detected.",
        examples=["Face appears happy but voice indicates anger."],
    )


# ── UNIFIED RESPONSE MODEL ────────────────────────────────────────────────────

class UnifiedEmotionResponse(BaseModel):
    """
    The payload every mode emits — text, voice, multimodal and live stream all
    return this identical shape, so the LLM side needs only one parser.

    See contract/CONTRACT.md for the full field reference, and
    tests/test_contract.py for the invariants this shape guarantees.
    """
    session_id: str
    user_input: UserInput
    emotion_analysis: EmotionAnalysis
    tone_analysis: ToneAnalysis
    conflict_analysis: ConflictAnalysis

    class Config:
        extra = "allow"
        json_schema_extra = {
            # A real payload: someone saying "It is fine, really." with an angry
            # voice and a smiling face. Note that `happy` scores higher than
            # `angry` while dominant_emotion is `angry` — always trust
            # dominant_emotion, never the argmax of the probabilities.
            "example": {
                "session_id": "sess-mock-conflict",
                "user_input": {
                    "text": "It is fine, really.",
                    "timestamp": "2026-08-22T07:48:10Z"
                },
                "emotion_analysis": {
                    "dominant_emotion": "angry",
                    "confidence": 0.29,
                    "emotion_probabilities": {
                        "happy": 0.48, "sad": 0.01, "angry": 0.39,
                        "surprised": 0.01, "fear": 0.01, "disgust": 0.01,
                        "neutral": 0.01, "empathetic": 0.01, "concerned": 0.01,
                        "shame": 0.01, "guilt": 0.01, "anxiety": 0.01,
                        "frustration": 0.01, "joy": 0.01, "calm": 0.01
                    }
                },
                "tone_analysis": {
                    "tone": "hostile",
                    "confidence": 0.82
                },
                "conflict_analysis": {
                    "detected": True,
                    "type": "masked_anger",
                    "details": "Face appears happy but voice indicates anger."
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