"""
src/interactive_modes.py
========================
Analysis steps shared by the API routes.

Everything here works on a FILE that arrived from a client. Nothing captures
from a microphone or a camera — the server owns no capture devices, which is
what lets many people use it at once.
"""

import sys
import os
import threading

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from src.ser.ser_engine import SEREngine
except ImportError:
    SEREngine = None

try:
    from src.text_emotion.analysis import analyze_text_emotion, load_emotion_model
except ImportError:
    def analyze_text_emotion(text, threshold=0.1): return []
    def load_emotion_model(): return None

from src.streaming.unified_pipeline import build_text_state, build_voice_state


# ── Whisper serialisation ─────────────────────────────────────────────────────
# openai-whisper's decoder is NOT safe to call concurrently on one model
# object, and registry.get("whisper") hands every caller the same object.
#
# DecodingTask installs PyTorch forward hooks on the shared decoder's key/value
# projections to build its kv-cache, and those hooks return a value, which
# PyTorch treats as replacing the module's output. Two decodes running at once
# therefore write into each other's cache: request A's hook fires inside
# request B's forward pass and hands back A's accumulation. Both run greedy
# decoding at batch size 1 and the causal mask broadcasts, so nothing raises —
# you get a plausible transcript containing fragments of the other person's
# speech, which then feeds the text emotion model and is forwarded downstream.
# A wrong answer, a privacy leak between users, and completely silent.
#
# Two simultaneous requests are enough, and /analyze/voice plus /analyze/video
# counts as two because video calls this same pipeline.
#
# Serialising costs no throughput: Whisper base already saturates the cores, so
# a second concurrent decode was never running in parallel in any real sense.
# The live call is unaffected and does not take this lock — it uses
# faster-whisper, whose CTranslate2 backend serialises internally.
_WHISPER_LOCK = threading.Lock()


# ── Whisper hallucination patterns ────────────────────────────────────────────
# Whisper produces these on silence, background noise, or very short audio.
_WHISPER_HALLUCINATIONS = {
    "", " ", "...", ". . .", "....", " ...",
    "you", "the", "a",
    "thank you", "thanks",
    "bye", "goodbye",
    "hmm", "mm", "um",
}


def _is_hallucination(text: str) -> bool:
    """
    Returns True if Whisper output is a known hallucination pattern
    and should be treated as empty transcription.
    """
    if not text:
        return True
    cleaned = text.strip().lower()
    # All dots or all spaces
    if all(c in ".… " for c in cleaned):
        return True
    # Known bad outputs
    if cleaned in _WHISPER_HALLUCINATIONS:
        return True
    # Single or two character outputs are always noise
    if len(cleaned) <= 2:
        return True
    return False


def _check_audio_has_speech(
    wav_path: str,
    silence_threshold: float = None,
) -> bool:
    """
    Quick RMS energy check on a WAV file before sending it to Whisper.
    Returns False if the file is silent or near-silent.

    Prevents:
        1. Whisper hallucinating "..." on silence
        2. SER reading "angry" out of microphone background noise

    The threshold is the SAME constant the live call uses as its absolute
    floor. It used to be a separate hardcoded 0.01 here, which meant the two
    paths disagreed about what counts as silence: measuring the recordings in
    data/recordings/, five of eighteen sit entirely below 0.01, so the same
    quiet speech was rejected outright with a 422 on this path while the live
    call analysed it fine.
    """
    if silence_threshold is None:
        from src.streaming.turn_detector import MIN_ABSOLUTE_THRESHOLD
        silence_threshold = MIN_ABSOLUTE_THRESHOLD

    try:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        data     = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = np.sqrt(np.mean(data ** 2))
        return rms > silence_threshold
    except Exception:
        # If the check fails for any reason, let Whisper try anyway.
        return True


# ── Pure logic functions ──────────────────────────────────────────────────────

def process_text_emotion(text: str) -> dict:
    """
    Analyzes text for emotion and returns built text_state.
    load_emotion_model() is a no-op — registry handles loading.
    """
    load_emotion_model()
    results    = analyze_text_emotion(text, threshold=0.05)
    text_state = build_text_state(text, results)
    return text_state


def process_voice_pipeline(wav_path: str):
    """
    Runs SER + Whisper STT + Text Emotion on a given audio file.
    All models obtained from ModelRegistry — no loading here.

    Includes:
        - Pre-Whisper silence check to avoid hallucinations
        - Post-Whisper hallucination filtering

    Returns:
        tuple: (text_state, voice_state, stt_result, ser_result)
               voice_state["confidence"] is the real SpeechBrain softmax score.
    """
    if not SEREngine or not wav_path or not os.path.exists(wav_path):
        return None, None, "N/A", "N/A"

    import concurrent.futures

    # ── SER and STT run CONCURRENTLY ─────────────────────────────────────────
    # They read the same file and share nothing, so ordering never mattered —
    # it was only ever sequential by accident of how the code was written.
    # Both are C-level calls that release the GIL, so this is a real saving.
    #
    # This is purely a scheduling change: identical inputs, identical models,
    # identical outputs. Verified by comparing results before and after across
    # the recordings in data/recordings/.

    def _run_ser():
        try:
            engine = SEREngine()
            label, confidence = engine.predict_emotion(wav_path)
            print(f"[VoicePipeline] SER: {label} (conf={confidence:.3f})")
            return label, confidence
        except Exception as e:
            print(f"[VoicePipeline] SER failed: {e}")
            return "N/A", 0.0

    def _run_stt():
        # The silence gate stays INSIDE this branch so it still runs before
        # Whisper — it exists to stop Whisper hallucinating on silence.
        if not _check_audio_has_speech(wav_path):
            print("[VoicePipeline] ⚠️  Audio appears silent — skipping Whisper.")
            return "N/A"
        try:
            from src.core.model_registry import registry
            model = registry.get("whisper")
            with _WHISPER_LOCK:
                transcription = model.transcribe(wav_path)
            raw_text = transcription["text"].strip()

            if _is_hallucination(raw_text):
                print(f"[VoicePipeline] ⚠️  Hallucination filtered: '{raw_text}'")
                return "N/A"

            print(f"[VoicePipeline] STT: '{raw_text}'")
            return raw_text
        except Exception as e:
            print(f"[VoicePipeline] STT failed: {e}")
            return "N/A"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_ser = pool.submit(_run_ser)
        f_stt = pool.submit(_run_stt)
        ser_result, ser_confidence = f_ser.result()
        stt_result = f_stt.result()

    # ── Step 4: Text Emotion ──────────────────────────────────────────────────
    text_state = None
    if stt_result and stt_result != "N/A":
        try:
            te_results = analyze_text_emotion(stt_result, threshold=0.05)
            text_state = build_text_state(stt_result, te_results)
        except Exception as e:
            print(f"[VoicePipeline] Text emotion failed: {e}")

    # ── Voice State ───────────────────────────────────────────────────────────
    # Built by the shared helper so the upload paths and the live call cannot
    # drift apart on how confidence becomes reliability.
    voice_state = build_voice_state(ser_result, ser_confidence)
    if voice_state:
        print(f"[VoicePipeline] voice_state → emotion={ser_result}, "
              f"conf={ser_confidence:.3f}, "
              f"reliability={voice_state['reliability']}")

    return text_state, voice_state, stt_result, ser_result
