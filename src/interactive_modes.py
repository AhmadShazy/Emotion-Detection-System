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
    print(f"Warning: Could not import SER Engine: {e}")
    SEREngine = None

try:
    from src.ser.recorder import record_audio
except ImportError as e:
    print(f"Warning: Could not import audio recorder: {e}")
    record_audio = None

try:
    import whisper
except ImportError as e:
    print(f"Warning: Could not import Whisper: {e}")
    whisper = None

try:
    from src.text_emotion.analysis import analyze_text_emotion, load_emotion_model
except ImportError as e:
    print(f"Warning: Could not import Text Emotion module: {e}")
    def analyze_text_emotion(text, threshold=0.1): return []
    def load_emotion_model(): return None

try:
    from src.faceexpression.classifier import analyze_openface_csv
except ImportError as e:
    print(f"Warning: Could not import OpenFace classifier: {e}")
    def analyze_openface_csv(csv_path):
        print("❌ OpenFace classifier not available.")
        return None

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError as e:
    print(f"Warning: Could not import sounddevice/scipy: {e}")
    sd = None
    wav_write = None

from src.streaming.unified_pipeline import build_text_state, process_and_print_unified_json


# ===============================
# OPTION 1 — Text Emotion Analysis
# ===============================

def run_text_emotion_input():
    print("\n==========================================")
    print("💬 TEXT EMOTION ANALYSIS")
    print("==========================================")
    print("Type your text below and press Enter.")
    print("(Type 'quit' to return to main menu)\n")

    # Pre-warm model if not already loaded
    load_emotion_model()

    while True:
        try:
            text = input("📝 Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n🛑 Cancelled.")
            break

        if text.lower() in ("quit", "exit", "q"):
            break

        if not text:
            print("⚠️  Please enter some text.\n")
            continue

        print("\n🧠 Analyzing emotion...")
        results = analyze_text_emotion(text, threshold=0.05)
        text_state = build_text_state(text, results)
        
        process_and_print_unified_json(
            text_state=text_state,
            voice_state=None,
            face_state=None,
            raw_text=text,
            voice_emo_raw="neutral",
            face_emo_raw="neutral"
        )


        again = input("Analyze another? (Enter to continue / 'q' to go back): ").strip().lower()
        if again in ("q", "quit", "exit"):
            break
        print()


# ===============================
# OPTION 2 — Voice Analysis (Combined)
# ===============================

def run_voice_combined_pipeline(wav_path=None):
    """
    Runs SER + Whisper STT + Text Emotion on audio.
    If wav_path is None  → records a fresh 10-second clip first.
    If wav_path is given → skips recording and reuses that file.
    Returns (text_state, voice_state, stt_result, ser_result) for callers that need the data.
    """
    standalone = (wav_path is None)  # True when called directly from Option 2

    if standalone:
        print("\n==========================================")
        print("🎤 VOICE ANALYSIS")
        print("==========================================")
        print("This will record your voice ONCE and run:")
        print("  • Speech Emotion Recognition (SER)")
        print("  • Speech-to-Text (Whisper)")
        print("  • Text Emotion Analysis (RoBERTa)")

    if not SEREngine:
        print("❌ SER Engine not available.")
        return "N/A", "N/A", "N/A"
    if not whisper:
        print("❌ Whisper not available.")
        return "N/A", "N/A", "N/A"

    # ── Step 1: Record Audio (only if no file was passed in) ──
    if wav_path is None:
        if not record_audio:
            print("❌ Audio recorder not available.")
            return "N/A", "N/A", "N/A"

        DATA_DIR = os.path.join(SCRIPT_DIR, "data", "recordings")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wav_path = os.path.join(DATA_DIR, f"voice_analysis_{timestamp}.wav")

        print(f"\n🎙️  Recording for 10 seconds...")
        print("👉  Speak now!\n")

        try:
            record_audio(duration=10, filename=wav_path)
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return "N/A", "N/A", "N/A"

        print(f"\n✅ Audio saved: {wav_path}")
        print("-" * 50)
    else:
        print(f"\n🔁 Reusing recorded audio: {wav_path}")
        print("-" * 50)

    # ── Step 2: SER ──
    ser_result = "N/A"
    print("\n🧠 Running Speech Emotion Recognition...")
    try:
        engine = SEREngine()
        ser_result = engine.predict_emotion(wav_path)
    except Exception as e:
        print(f"⚠️  SER failed: {e}")

    # ── Step 3: Whisper STT ──
    stt_result = "N/A"
    print("\n📖 Transcribing with Whisper...")
    try:
        whisper_cache = os.path.join(SCRIPT_DIR, "external", "whisper")
        if not os.path.exists(whisper_cache):
            os.makedirs(whisper_cache)
        model = whisper.load_model("base", download_root=whisper_cache)
        transcription = model.transcribe(wav_path)
        stt_result = transcription["text"].strip()
    except Exception as e:
        print(f"⚠️  Transcription failed: {e}")

    # ── Step 4: Text Emotion ──
    text_state = None
    if stt_result and stt_result != "N/A":
        print("\n💬 Analyzing Text Emotion...")
        try:
            te_results = analyze_text_emotion(stt_result, threshold=0.05)
            text_state = build_text_state(stt_result, te_results)
        except Exception as e:
            print(f"⚠️  Text emotion analysis failed: {e}")

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

    # ── Final Report (only when called standalone from Option 2) ──
    if standalone:
        process_and_print_unified_json(
            text_state=text_state,
            voice_state=voice_state,
            face_state=None,
            raw_text=stt_result if stt_result != "N/A" else "",
            voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
            face_emo_raw="neutral"
        )
        return

    return text_state, voice_state, stt_result, ser_result


# ===============================
# OPTION 3 — Multimodal Recording (Face + Voice)
# ===============================

def run_multimodal_recording():
    print("\n==========================================")
    print("🎥 MULTIMODAL RECORDING")
    print("==========================================")
    print("This will simultaneously record:")
    print("  • 🎥 Face expressions (OpenFace)")
    print("  • 🎤 Audio (Microphone)")
    print("\nThen analyse:")
    print("  • Face Emotion (AU-based classifier)")
    print("  • Voice Emotion (SER / Wav2Vec2)")
    print("  • Speech-to-Text (Whisper)")
    print("  • Text Emotion (RoBERTa)")

    if sd is None or wav_write is None:
        print("❌ sounddevice / scipy not available. Cannot record audio.")
        return
    if not SEREngine:
        print("❌ SER Engine not available.")
        return
    if not whisper:
        print("❌ Whisper not available.")
        return

    # ── Paths ──
    OPENFACE_DIR = os.path.join(SCRIPT_DIR, "external", "openface", "OpenFace_2.2.0_win_x64")
    OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
    OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "data", "processed")
    DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "recordings")

    for d in (OUTPUT_DIR, DATA_DIR):
        if not os.path.exists(d):
            os.makedirs(d)

    timestamp        = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    of_filename      = f"multimodal_{timestamp}"
    csv_path         = os.path.join(OUTPUT_DIR, f"{of_filename}.csv")
    wav_path         = os.path.join(DATA_DIR,   f"multimodal_{timestamp}.wav")

    # ── Check OpenFace binary exists ──
    if not os.path.exists(OPENFACE_EXE):
        print(f"❌ OpenFace executable not found at: {OPENFACE_EXE}")
        print("   Skipping face recording. Only audio will be analysed.")
        face_available = False
    else:
        face_available = True

    # ── Audio recording state ──
    FS           = 16000
    audio_chunks = []
    stop_event   = threading.Event()

    def _audio_worker():
        """Continuously records 0.5-second chunks until stop_event is set."""
        with sd.InputStream(samplerate=FS, channels=1, dtype='int16') as stream:
            while not stop_event.is_set():
                chunk, _ = stream.read(FS // 2)   # 0.5 s chunks
                audio_chunks.append(chunk.copy())

    # ── 1. Launch OpenFace (non-blocking) ──
    of_process = None
    if face_available:
        of_cmd = [
            OPENFACE_EXE,
            "-device", "0",
            "-out_dir", OUTPUT_DIR,
            "-of",     of_filename
        ]
        try:
            of_process = subprocess.Popen(of_cmd, cwd=OPENFACE_DIR,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            print("\n▶  OpenFace launched (non-blocking).")
        except FileNotFoundError:
            print("⚠️  Could not launch OpenFace. Face analysis will be skipped.")
            face_available = False

    # ── 2. Start audio recording thread ──
    audio_thread = threading.Thread(target=_audio_worker, daemon=True)
    audio_thread.start()

    # ── 3. Wait for user to stop ──
    print("\n🔴 RECORDING — press Enter to stop...")
    start_time = time.time()
    try:
        input()                        # blocks until Enter
    except (KeyboardInterrupt, EOFError):
        pass
    elapsed = time.time() - start_time
    print(f"\n⏱  Recorded {elapsed:.1f} s.")

    # ── 4. Stop both streams ──
    stop_event.set()
    audio_thread.join(timeout=3)

    if of_process is not None and of_process.poll() is None:
        of_process.terminate()
        try:
            of_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            of_process.kill()
        print("✅ OpenFace stopped.")

    # ── 5. Save WAV ──
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
        wav_write(wav_path, FS, audio_data)
        print(f"✅ Audio saved: {wav_path}")
    else:
        print("⚠️  No audio captured.")
        wav_path = None

    # ── 6. Post-processing ──
    print("\n" + "="*60)
    print("⚙️  POST-PROCESSING — please wait...")
    print("="*60)

    # 6a. Voice pipeline (reuse Option 2, skip recording)
    ser_result      = "N/A"
    stt_result      = "N/A"
    text_state      = None
    voice_state     = None
    if wav_path and os.path.exists(wav_path):
        text_state, voice_state, stt_result, ser_result = run_voice_combined_pipeline(wav_path=wav_path)

    # 6b. Face analysis
    face_timeline = "N/A"
    face_state = None
    if face_available:
        time.sleep(1)   # short buffer for OpenFace to flush CSV
        if os.path.exists(csv_path):
            print("\n🙂 Analysing face expressions...")
            face_timeline, face_state = analyze_openface_csv(csv_path)
            if not face_timeline:
                face_timeline = "No valid face frames detected."
        else:
            print(f"⚠️  OpenFace CSV not found: {csv_path}")
            face_timeline = "CSV not generated."

    # ── 7. Combined Report ──
    face_emo_raw = face_state["emotion"] if face_state else "neutral"
    
    process_and_print_unified_json(
        text_state=text_state,
        voice_state=voice_state,
        face_state=face_state,
        raw_text=stt_result if stt_result != "N/A" else "",
        voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
        face_emo_raw=face_emo_raw
    )
