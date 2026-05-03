"""
schemas/emotion.py

Pydantic models that mirror the exact JSON shapes flowing through
src/streaming/llm_adapter.py  →  src/streaming/unified_pipeline.py

REQUEST models  → what the API receives from the client
RESPONSE models → what the API returns to the client
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============================================================
# REQUEST MODELS
# ============================================================

class TextAnalysisRequest(BaseModel):
    """
    Body for:  POST /analyze/text
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The raw text to analyze for emotion.",
        examples=["I feel really happy today!"],
    )

    class Config:
        json_schema_extra = {
            "example": {"text": "I feel really happy today!"}
        }


class MultimodalStartRequest(BaseModel):
    """
    Body for:  POST /analyze/multimodal/start
    No fields required — kept as a model for future extensibility.
    """
    pass


class MultimodalStopRequest(BaseModel):
    """
    Body for:  POST /analyze/multimodal/stop
    """
    session_id: str = Field(
        ...,
        description="The session_id returned by /analyze/multimodal/start.",
        examples=["mm-3f7a1c2b"],
    )


# ============================================================
# NESTED SUB-MODELS  (building blocks for the response)
# ============================================================

class UserInput(BaseModel):
    """The raw input captured during the analysis turn."""
    text: str = Field(description="Transcribed or submitted text.")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of the analysis.")


class EmotionAnalysis(BaseModel):
    """Fused emotion output from EmotionStateManager + LLMAdapter."""
    dominant_emotion: str = Field(description="The top emotion after fusion.")
    confidence: float = Field(ge=0.0, le=1.0)
    emotion_probabilities: Dict[str, float] = Field(
        description="Probability distribution across the 15 LLM emotion classes."
    )


class ToneAnalysis(BaseModel):
    """How the user spoke, separate from what they felt."""
    tone: str = Field(description="e.g. 'cheerful', 'somber', 'measured'.")
    confidence: float = Field(ge=0.0, le=1.0)


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: str = Field(description="'user' or 'assistant'.")
    content: str = Field(description="The spoken/typed text of this turn.")


class ConversationContextBlock(BaseModel):
    """Sliding window of recent conversation turns for LLM context."""
    window_size: int = Field(description="Max turns retained (always 6).")
    turns: List[Dict[str, Any]] = Field(description="Full turn records with emotion/tone metadata.")


class ContextBlock(BaseModel):
    """Short-form conversation history (role + content only)."""
    conversation_history: List[ConversationTurn]


# ============================================================
# UNIFIED RESPONSE MODEL
# ============================================================

class UnifiedEmotionResponse(BaseModel):
    """
    The canonical V2 payload produced by process_and_print_unified_json().
    All four API modes (text, voice, multimodal, stream) return this shape.

    Maps 1-to-1 to the dict returned by LLMAdapter.process().
    """
    session_id: str = Field(description="Shared session ID (persists across turns).")
    user_input: UserInput
    emotion_analysis: EmotionAnalysis
    tone_analysis: ToneAnalysis
    context: ContextBlock
    conversation_context: ConversationContextBlock

    class Config:
        # Allow extra fields from the pipeline without raising validation errors.
        # This future-proofs the schema as llm_adapter.py evolves.
        extra = "allow"
        json_schema_extra = {
            "example": {
                "session_id": "sess-a1b2c3d4",
                "user_input": {
                    "text": "I feel really happy today!",
                    "timestamp": "2026-05-03T00:00:00Z"
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
                },
                "context": {
                    "conversation_history": []
                },
                "conversation_context": {
                    "window_size": 6,
                    "turns": []
                }
            }
        }


# Explicit alias for clarity in voice router
VoiceAnalysisResponse = UnifiedEmotionResponse


# ============================================================
# MULTIMODAL SESSION MODELS
# ============================================================

class MultimodalSessionStarted(BaseModel):
    """
    Returned immediately after POST /analyze/multimodal/start.
    The client stores session_id and uses it to call /stop.
    """
    session_id: str = Field(description="Unique ID for this recording session.")
    status: str = Field(default="recording", description="Always 'recording' on success.")
    max_duration_seconds: int = Field(default=30, description="Server-enforced hard cap.")
    message: str = Field(
        default="Recording started. Call POST /analyze/multimodal/stop when done.",
        description="Human-readable next-step instruction.",
    )


# ============================================================
# ERROR MODEL
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error shape returned on 4xx / 5xx responses."""
    detail: str = Field(description="Human-readable error description.")
    code: Optional[str] = Field(default=None, description="Optional machine-readable error code.")
