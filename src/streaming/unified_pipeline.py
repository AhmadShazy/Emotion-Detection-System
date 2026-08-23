"""
src/streaming/unified_pipeline.py
==================================
Central emotion pipeline with thread-safe per-session state.

Conversation history removed — context window is managed by LLM side.
Sessions are kept for:
  - Per-user isolated EmotionStateManager (emotion smoothing)
  - Per-user isolated LLMAdapter
  - Thread safety across concurrent requests
"""

import json
import uuid
import datetime
import threading
import time

from src.core.config import LOG_PAYLOADS
from src.streaming.emotion_state_manager import EmotionStateManager
from src.streaming.llm_adapter import LLMAdapter

# ── Session configuration ─────────────────────────────────────────────────────
SESSION_TIMEOUT_MINUTES = 30

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict   = {}
_sessions_lock    = threading.Lock()


# ── Session object ────────────────────────────────────────────────────────────

class _Session:
    """
    Holds per-user isolated state.
    Conversation history removed — LLM handles context on its side.
    """
    def __init__(self, session_id: str):
        self.session_id  = session_id
        self.created_at  = time.time()
        self.last_active = time.time()
        self.state_manager = EmotionStateManager()
        self.llm_adapter   = LLMAdapter()
        self.lock          = threading.Lock()

    def touch(self):
        self.last_active = time.time()

    def is_expired(self):
        idle_seconds = time.time() - self.last_active
        return idle_seconds > SESSION_TIMEOUT_MINUTES * 60


# ── Session manager helpers ───────────────────────────────────────────────────

def get_or_create_session(session_id: str | None = None) -> _Session:
    with _sessions_lock:
        if session_id is None or session_id not in _sessions:
            new_id  = session_id or f"sess-{uuid.uuid4().hex[:8]}"
            session = _Session(new_id)
            _sessions[new_id] = session
            return session

        session = _sessions[session_id]
        session.touch()
        return session


def close_session(session_id: str) -> bool:
    with _sessions_lock:
        return _sessions.pop(session_id, None) is not None


def active_session_count() -> int:
    with _sessions_lock:
        return len(_sessions)


# ── Background expiry thread ──────────────────────────────────────────────────

def _expiry_worker():
    while True:
        time.sleep(5 * 60)
        with _sessions_lock:
            expired = [sid for sid, s in _sessions.items() if s.is_expired()]
            for sid in expired:
                del _sessions[sid]
                print(f"[Session] ⏱  Expired session removed: {sid}")


_expiry_thread = threading.Thread(
    target=_expiry_worker, daemon=True, name="session-expiry"
)
_expiry_thread.start()


# ── Text state builder ────────────────────────────────────────────────────────

def build_voice_state(emotion: str, confidence: float) -> dict | None:
    """
    Builds the voice modality dict the fusion engine consumes.

    Shared by every path that runs speech emotion recognition. It existed in two
    places before — once for the upload paths and once for the live call — with
    the reliability formula written out separately in each. Identical today, but
    that is exactly the shape of thing that drifts: a tweak in one copy silently
    makes the same audio score differently depending on which mode it arrived
    through.

    The +0.15 reliability boost is deliberate: SpeechBrain's top-class softmax
    sits around 0.6-0.8 even on clean speech, so using it raw would understate
    how much the voice signal deserves to count.
    """
    if not emotion or emotion == "N/A":
        return None

    reliability = min(1.0, confidence + 0.15)
    return {
        "source":          "voice",
        "emotion":         emotion,
        "confidence":      round(confidence, 4),
        "average_emotion": emotion,
        "peak_emotion":    emotion,
        "reliability":     round(reliability, 4),
    }


def build_text_state(text: str, te_results: list) -> dict | None:
    if not text or text == "N/A":
        return None

    text_state = {"emotion": "Neutral", "confidence": 0.0, "reliability": 1.0}

    if te_results:
        top        = te_results[0]
        words      = len(text.split())
        reliability = 1.0
        if words < 3:
            reliability -= 0.3
        if "!" in text or "?" in text:
            reliability += 0.2
        reliability = min(1.0, max(0.0, reliability))

        text_state = {
            "emotion":     top["label"],
            "confidence":  top["score"],
            "reliability": reliability,
        }

    return text_state


# ── Core pipeline function ────────────────────────────────────────────────────

def process_and_print_unified_json(
    text_state,
    voice_state,
    face_state,
    raw_text:      str,
    voice_emo_raw: str,
    face_emo_raw:  str,
    session_id:    str | None = None,
    on_payload=None,
) -> dict:
    """
    Routes emotion states through the per-session pipeline and returns
    the final V2 JSON payload.

    Conversation history is NOT stored here.
    Context window management is handled by the LLM side.
    """
    # ── 1. Get or create isolated session ─────────────────────────────────────
    session = get_or_create_session(session_id)

    with session.lock:

        # ── 2. Fusion ─────────────────────────────────────────────────────────
        unified_emotion_data = session.state_manager.fuse(
            text_state, voice_state, face_state
        )

        # ── 3. Build raw inputs ───────────────────────────────────────────────
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        raw_inputs = {
            "text":          raw_text,
            "timestamp":     now_str,
            "voice_emotion": voice_emo_raw,
            "face_emotion":  face_emo_raw,
        }

        context = {
            "session_id": session.session_id,
        }

        # ── 4. LLM adapter ────────────────────────────────────────────────────
        payload = session.llm_adapter.process(
            fusion_output=unified_emotion_data,
            raw_inputs=raw_inputs,
            context=context,
        )

    # ── 5. Server-side log ────────────────────────────────────────────────────
    # One line by default. The full payload contains the user's transcribed
    # speech, so dumping it on every request writes what people said into the
    # server log — fine on a laptop, wrong for a hosted service. Set
    # LOG_PAYLOADS=true to get the full dump back while developing.
    emotion = payload["emotion_analysis"]
    conflict = payload.get("conflict_analysis", {})
    print(
        f"[Payload] session={payload['session_id']} "
        f"emotion={emotion['dominant_emotion']} "
        f"conf={emotion['confidence']} "
        f"tone={payload['tone_analysis']['tone']}"
        + (f" conflict={conflict['type']}" if conflict.get("detected") else "")
    )

    if LOG_PAYLOADS:
        print(json.dumps(payload, indent=2))

    # ── 6. Optional WebSocket push ────────────────────────────────────────────
    if on_payload is not None:
        on_payload(payload)

    return payload