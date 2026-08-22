import sys
import os
import time
import datetime
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

try:
    from src.faceexpression.classifier import analyze_openface_csv
except ImportError:
    def analyze_openface_csv(csv_path): return None

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError:
    sd        = None
    wav_write = None

from src.streaming.unified_pipeline import build_text_state


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
    silence_threshold: float = 0.01,
) -> bool:
    """
    Quick RMS energy check on a WAV file before sending to Whisper.
    Returns False if the file is silent or near-silent.

    Prevents:
        1. Whisper hallucinating "..." on silence
        2. SER detecting "angry" from mic background noise
    """
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        data     = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = np.sqrt(np.mean(data ** 2))
        return rms > silence_threshold
    except Exception:
        # If check fails for any reason let Whisper try anyway
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

    # ── Step 1: SER ───────────────────────────────────────────────────────────
    ser_result     = "N/A"
    ser_confidence = 0.0   # safe default — overwritten on success
    try:
        engine                  = SEREngine()
        ser_result, ser_confidence = engine.predict_emotion(wav_path)
        print(f"[VoicePipeline] SER: {ser_result} (conf={ser_confidence:.3f})")
    except Exception as e:
        print(f"[VoicePipeline] SER failed: {e}")

    # ── Step 2: Pre-check audio energy before Whisper ─────────────────────────
    stt_result = "N/A"

    if not _check_audio_has_speech(wav_path):
        print("[VoicePipeline] ⚠️  Audio appears silent — skipping Whisper.")
    else:
        # ── Step 3: Whisper STT ───────────────────────────────────────────────
        try:
            from src.core.model_registry import registry
            model         = registry.get("whisper")
            transcription = model.transcribe(wav_path)
            raw_text      = transcription["text"].strip()

            if _is_hallucination(raw_text):
                print(f"[VoicePipeline] ⚠️  Hallucination filtered: '{raw_text}'")
                stt_result = "N/A"
            else:
                stt_result = raw_text
                print(f"[VoicePipeline] STT: '{stt_result}'")

        except Exception as e:
            print(f"[VoicePipeline] STT failed: {e}")

    # ── Step 4: Text Emotion ──────────────────────────────────────────────────
    text_state = None
    if stt_result and stt_result != "N/A":
        try:
            te_results = analyze_text_emotion(stt_result, threshold=0.05)
            text_state = build_text_state(stt_result, te_results)
        except Exception as e:
            print(f"[VoicePipeline] Text emotion failed: {e}")

    # ── Voice State ───────────────────────────────────────────────────────────
    # Reliability: derived from real confidence — low confidence = less reliable.
    # Capped at 1.0. A small boost (+0.15) is applied because SpeechBrain's
    # top-class softmax scores often sit around 0.6–0.8 on clean speech.
    voice_state = None
    if ser_result and ser_result != "N/A":
        reliability = min(1.0, ser_confidence + 0.15)
        voice_state = {
            "source":          "voice",
            "emotion":         ser_result,
            "confidence":      round(ser_confidence, 4),
            "average_emotion": ser_result,
            "peak_emotion":    ser_result,
            "reliability":     round(reliability, 4),
        }
        print(f"[VoicePipeline] voice_state → emotion={ser_result}, "
              f"conf={ser_confidence:.3f}, reliability={reliability:.3f}")

    return text_state, voice_state, stt_result, ser_result


def process_multimodal_data(
    wav_path:       str,
    csv_path:       str,
    face_available: bool = True,
):
    """
    Processes audio and face tracking data together.
    Uses the same hallucination-filtered voice pipeline.

    Returns:
        tuple: (text_state, voice_state, face_state,
                stt_result, ser_result, face_timeline)
    """
    text_state  = None
    voice_state = None
    stt_result  = "N/A"
    ser_result  = "N/A"

    if wav_path and os.path.exists(wav_path):
        text_state, voice_state, stt_result, ser_result = (
            process_voice_pipeline(wav_path)
        )

    face_timeline = "N/A"
    face_state    = None

    if face_available:
        time.sleep(1)   # short buffer for OpenFace to flush CSV
        if csv_path and os.path.exists(csv_path):
            result = analyze_openface_csv(csv_path)
            if result:
                face_timeline, face_state = result
                if not face_timeline:
                    face_timeline = "No valid face frames detected."
        else:
            face_timeline = "CSV not generated."

    return (
        text_state, voice_state, face_state,
        stt_result, ser_result, face_timeline,
    )