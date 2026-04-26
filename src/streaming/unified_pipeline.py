import json
import uuid
import datetime

from src.streaming.emotion_state_manager import EmotionStateManager
from src.streaming.llm_adapter import LLMAdapter

# Global session and history state for consistent JSON payload context across menu options
GLOBAL_SESSION_ID = f"sess-{uuid.uuid4().hex[:8]}"
GLOBAL_CONVERSATION_HISTORY = []

_state_manager = None
_llm_adapter = None

def get_pipeline():
    global _state_manager, _llm_adapter
    if _state_manager is None:
        _state_manager = EmotionStateManager()
    if _llm_adapter is None:
        _llm_adapter = LLMAdapter()
    return _state_manager, _llm_adapter

def build_text_state(text, te_results):
    """
    Standardizes the text reliability logic to generate a consistent text_state snapshot.
    """
    if not text or text == "N/A":
        return None
        
    text_state = {"emotion": "Neutral", "confidence": 0.0, "reliability": 1.0}
    if te_results and len(te_results) > 0:
        top_emotion = te_results[0]['label']
        top_score = te_results[0]['score']
        
        # Calculate Reliability based on length and punctuation
        text_reliability = 1.0
        words = len(text.split())
        if words < 3:
            text_reliability -= 0.3
        if "!" in text or "?" in text:
            text_reliability += 0.2
        text_reliability = min(1.0, max(0.0, text_reliability))
        
        text_state = {"emotion": top_emotion, "confidence": top_score, "reliability": text_reliability}
        
    return text_state

def process_and_print_unified_json(text_state, voice_state, face_state, raw_text, voice_emo_raw, face_emo_raw):
    """
    Routes given states through the central EmotionStateManager and LLMAdapter,
    updates the conversation context, and prints the final V2 JSON payload.
    """
    state_manager, llm_adapter = get_pipeline()
    
    # 1. Decision-level Fusion
    unified_emotion_data = state_manager.fuse(text_state, voice_state, face_state)
    
    # 2. Build Inputs for LLM Adapter
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_inputs = {
        "text": raw_text,
        "timestamp": now_str,
        "voice_emotion": voice_emo_raw,
        "face_emotion": face_emo_raw
    }
    
    context = {
        "session_id": GLOBAL_SESSION_ID,
        "conversation_history": [
            {"role": t["speaker"], "content": t["text"]}
            for t in GLOBAL_CONVERSATION_HISTORY
        ],
        "turns": GLOBAL_CONVERSATION_HISTORY
    }
    
    # 3. Process via LLM Adapter
    payload = llm_adapter.process(
        fusion_output=unified_emotion_data,
        raw_inputs=raw_inputs,
        context=context
    )
    
    # 4. Update Conversation History for subsequent turns
    turn_record = {
        "speaker": "user",
        "timestamp": now_str,
        "text": raw_text,
        "emotion": payload["emotion_analysis"]["dominant_emotion"],
        "tone": payload["tone_analysis"]["tone"]
    }
    GLOBAL_CONVERSATION_HISTORY.append(turn_record)
    if len(GLOBAL_CONVERSATION_HISTORY) > 6:
        GLOBAL_CONVERSATION_HISTORY.pop(0)
        
    # 5. Print the Output
    print("\n" + "="*80)
    print(">>> OUTBOUND V2 PAYLOAD")
    print("="*80)
    print(json.dumps(payload, indent=2))
    print("="*80)
    
    return payload
