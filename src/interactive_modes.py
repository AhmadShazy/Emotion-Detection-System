import sys
import os
import time
import datetime
import subprocess
import threading
import numpy as np

# Adjust SCRIPT_DIR to be the project root since this file is inside src/
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure project root is in python path
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from src.ser.ser_engine import SEREngine
except ImportError as e:
    SEREngine = None

try:
    from src.ser.recorder import record_audio
except ImportError as e:
    record_audio = None

try:
    import whisper
except ImportError as e:
    whisper = None

try:
    from src.text_emotion.analysis import analyze_text_emotion, load_emotion_model
except ImportError as e:
    def analyze_text_emotion(text, threshold=0.1): return []
    def load_emotion_model(): return None

try:
    from src.faceexpression.classifier import analyze_openface_csv
except ImportError as e:
    def analyze_openface_csv(csv_path): return None

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError as e:
    sd = None
    wav_write = None

from src.streaming.unified_pipeline import build_text_state

# ===============================
# PURE LOGIC FUNCTIONS
# ===============================

def process_text_emotion(text: str) -> dict:
    """
    Analyzes the text for emotion and returns the built text_state.
    """
    load_emotion_model()
    results = analyze_text_emotion(text, threshold=0.05)
    text_state = build_text_state(text, results)
    return text_state

def record_audio_clip(duration: int = 10) -> str:
    """
    Records an audio clip for a specified duration and returns the path to the saved WAV file.
    Returns None if recording fails or is unavailable.
    """
    if not record_audio:
        return None

    DATA_DIR = os.path.join(SCRIPT_DIR, "data", "recordings")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wav_path = os.path.join(DATA_DIR, f"voice_analysis_{timestamp}.wav")

    try:
        record_audio(duration=duration, filename=wav_path)
        return wav_path
    except Exception:
        return None

def process_voice_pipeline(wav_path: str):
    """
    Runs SER + Whisper STT + Text Emotion on a given audio file.
    Returns a tuple: (text_state, voice_state, stt_result, ser_result)
    """
    if not SEREngine or not whisper or not wav_path or not os.path.exists(wav_path):
        return None, None, "N/A", "N/A"

    # Step 1: SER
    ser_result = "N/A"
    try:
        engine = SEREngine()
        ser_result = engine.predict_emotion(wav_path)
    except Exception:
        pass

    # Step 2: Whisper STT
    stt_result = "N/A"
    try:
        whisper_cache = os.path.join(SCRIPT_DIR, "external", "whisper")
        if not os.path.exists(whisper_cache):
            os.makedirs(whisper_cache)
        model = whisper.load_model("base", download_root=whisper_cache)
        transcription = model.transcribe(wav_path)
        stt_result = transcription["text"].strip()
    except Exception:
        pass

    # Step 3: Text Emotion
    text_state = None
    if stt_result and stt_result != "N/A":
        try:
            te_results = analyze_text_emotion(stt_result, threshold=0.05)
            text_state = build_text_state(stt_result, te_results)
        except Exception:
            pass

    # Voice State
    voice_state = None
    if ser_result and ser_result != "N/A":
        voice_state = {
            "source": "voice",
            "emotion": ser_result,
            "confidence": 0.8,
            "average_emotion": ser_result,
            "peak_emotion": ser_result,
            "reliability": 1.0
        }

    return text_state, voice_state, stt_result, ser_result


def process_multimodal_data(wav_path: str, csv_path: str, face_available: bool = True):
    """
    Processes audio and face tracking data together.
    Returns a tuple: (text_state, voice_state, face_state, stt_result, ser_result, face_timeline)
    """
    text_state, voice_state, stt_result, ser_result = None, None, "N/A", "N/A"
    
    if wav_path and os.path.exists(wav_path):
        text_state, voice_state, stt_result, ser_result = process_voice_pipeline(wav_path)

    face_timeline = "N/A"
    face_state = None
    
    if face_available:
        time.sleep(1) # short buffer for OpenFace to flush CSV
        if csv_path and os.path.exists(csv_path):
            result = analyze_openface_csv(csv_path)
            if result:
                face_timeline, face_state = result
                if not face_timeline:
                    face_timeline = "No valid face frames detected."
        else:
            face_timeline = "CSV not generated."

    return text_state, voice_state, face_state, stt_result, ser_result, face_timeline
