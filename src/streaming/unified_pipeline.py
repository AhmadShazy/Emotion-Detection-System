"""
src/streaming/unified_pipeline.py
==================================
Central emotion pipeline with thread-safe per-session state.

Session Lifecycle (Option A — auto-create):
  - First request with a new session_id  → session created automatically
  - Each request updates last_active timestamp
  - Background thread expires sessions idle > SESSION_TIMEOUT_MINUTES
  - Session ID is returned in every payload so the client can reuse it

Thread Safety:
  - _sessions_lock guards the session store dict
  - Each session has its own Lock so concurrent requests for the
    SAME session queue up rather than corrupting shared state
  - Different sessions never block each other
"""

import json
import uuid
import datetime
import threading
import time

from src.streaming.emotion_state_manager import EmotionStateManager
from src.streaming.llm_adapter import LLMAdapter

# ── Session configuration ─────────────────────────────────────────────────────
SESSION_TIMEOUT_MINUTES = 30
SESSION_HISTORY_MAX     = 6       # sliding window kept per session

# ── Session store ─────────────────────────────────────────────────────────────
# { session_id: _Session }
_sessions: dict       = {}
_sessions_lock        = threading.Lock()  # guards the dict itself


# ── Session object ────────────────────────────────────────────────────────────

class _Session:
    """
    Holds all mutable state that belongs to one user/conversation.
    Each session is isolated — no shared state between sessions.
    """
    def __init__(self, session_id: str):
        self.session_id           = session_id
        self.created_at           = time.time()
        self.last_active          = time.time()
        self.conversation_history = []          # list of turn dicts
        self.state_manager        = EmotionStateManager()
        self.llm_adapter          = LLMAdapter()
        self.lock                 = threading.Lock()  # per-session concurrency guard

    def touch(self):
        """Updates last_active to now, preventing timeout expiry."""
        self.last_active = time.time()

    def is_expired(self):
        idle_seconds = time.time() - self.last_active
        return idle_seconds > SESSION_TIMEOUT_MINUTES * 60


# ── Session manager helpers ───────────────────────────────────────────────────

def get_or_create_session(session_id: str | None = None) -> _Session:
    """
    Returns an existing session or creates a new one.

    If session_id is None or unknown, a fresh session is created and its
    ID is returned inside the session object (auto-create pattern).
    """
    with _sessions_lock:
        # Auto-generate ID if none provided or if it's expired/unknown
        if session_id is None or session_id not in _sessions:
            new_id   = session_id or f"sess-{uuid.uuid4().hex[:8]}"
            session  = _Session(new_id)
            _sessions[new_id] = session
            return session

        session = _sessions[session_id]
        session.touch()
        return session


def close_session(session_id: str) -> bool:
    """Explicitly removes a session. Returns True if it existed."""
    with _sessions_lock:
        return _sessions.pop(session_id, None) is not None


def active_session_count() -> int:
    with _sessions_lock:
        return len(_sessions)


# ── Background expiry thread ──────────────────────────────────────────────────

def _expiry_worker():
    """
    Runs forever as a daemon thread.
    Every 5 minutes it sweeps the session store and removes expired sessions.
    """
    while True:
        time.sleep(5 * 60)  # check every 5 minutes
        with _sessions_lock:
            expired = [sid for sid, s in _sessions.items() if s.is_expired()]
            for sid in expired:
                del _sessions[sid]
                print(f"[Session] ⏱  Expired session removed: {sid}")


# Start expiry daemon once when module is first imported
_expiry_thread = threading.Thread(target=_expiry_worker, daemon=True, name="session-expiry")
_expiry_thread.start()


# ── Text state builder (shared utility) ──────────────────────────────────────

def build_text_state(text: str, te_results: list) -> dict | None:
    """
    Converts raw text + RoBERTa results into a standardised text_state dict
    consumed by EmotionStateManager.fuse().
    """
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

    Parameters
    ----------
    text_state, voice_state, face_state : dicts from respective workers
    raw_text       : original transcription / typed text
    voice_emo_raw  : raw SER label string
    face_emo_raw   : raw face classifier label string
    session_id     : optional — auto-created if None or unknown
    on_payload     : optional callback(dict) — used by WebSocket router
                     to push each turn's payload to the connected client.
                     CLI / live_orchestrator callers leave this as None.

    Returns
    -------
    dict — the full V2 payload (same shape as UnifiedEmotionResponse schema)
    """
    # ── 1. Get or create isolated session ─────────────────────────────────────
    session = get_or_create_session(session_id)

    # Per-session lock ensures concurrent requests for the SAME session
    # are serialised rather than corrupting shared state.
    with session.lock:

        # ── 2. Decision-level fusion ──────────────────────────────────────────
        unified_emotion_data = session.state_manager.fuse(
            text_state, voice_state, face_state
        )

        # ── 3. Build raw inputs for LLM adapter ───────────────────────────────
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        raw_inputs = {
            "text":          raw_text,
            "timestamp":     now_str,
            "voice_emotion": voice_emo_raw,
            "face_emotion":  face_emo_raw,
        }

        context = {
            "session_id": session.session_id,
            "conversation_history": [
                {"role": t["speaker"], "content": t["text"]}
                for t in session.conversation_history
            ],
            "turns": session.conversation_history,
        }

        # ── 4. LLM adapter processing ─────────────────────────────────────────
        payload = session.llm_adapter.process(
            fusion_output=unified_emotion_data,
            raw_inputs=raw_inputs,
            context=context,
        )

        # ── 5. Update this session's conversation history ─────────────────────
        turn_record = {
            "speaker":   "user",
            "timestamp": now_str,
            "text":      raw_text,
            "emotion":   payload["emotion_analysis"]["dominant_emotion"],
            "tone":      payload["tone_analysis"]["tone"],
        }
        session.conversation_history.append(turn_record)

        # Enforce sliding window
        if len(session.conversation_history) > SESSION_HISTORY_MAX:
            session.conversation_history.pop(0)

    # ── 6. Server-side log (harmless in API mode) ─────────────────────────────
    print("\n" + "=" * 80)
    print(">>> OUTBOUND V2 PAYLOAD")
    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80)

    # ── 7. Optional push callback (WebSocket router) ──────────────────────────
    if on_payload is not None:
        on_payload(payload)

    return payload