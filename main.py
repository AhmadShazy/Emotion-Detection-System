import sys
import os
import time
import datetime
import subprocess
import threading

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is in python path
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# ===============================
# IMPORTS — existing modules
# ===============================

try:
    from src.faceexpression.record_express import run_face_pipeline
except ImportError as e:
    print(f"Warning: Could not import Face Expression module: {e}")
    def run_face_pipeline():
        print("❌ Face Expression module not available.")

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
    from src.streaming.live_orchestrator import run_live_streaming_session
except ImportError as e:
    print(f"Warning: Could not import V2 Streaming module: {e}")
    def run_live_streaming_session():
        print("❌ V2 Streaming module not available.")


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

        print("\n" + "="*50)
        print("📊 TEXT EMOTION REPORT")
        print("="*50)

        if results:
            for i, res in enumerate(results[:5]):   # Show top 5
                bar = "█" * int(res['score'] * 20)
                print(f"  {'→' if i == 0 else '  '} {res['label']:<18} {res['score']:.2f}  {bar}")
        else:
            print("  → Could not determine emotion.")

        print("="*50)
        print()

        again = input("Analyze another? (Enter to continue / 'q' to go back): ").strip().lower()
        if again in ("q", "quit", "exit"):
            break
        print()


# ===============================
# OPTION 2 — Voice Analysis (Combined)
# ===============================

def run_voice_combined_pipeline(wav_path=None):
    """
    Runs SER + Whisper STT + Text Emotion on a voice recording.
    If wav_path is provided, skips recording and reuses that file.
    If wav_path is None, records 10 seconds from the microphone first.
    Returns a dict with keys: ser_result, stt_result, text_emotion_str
    """
    is_standalone = (wav_path is None)  # True when called directly from menu

    if is_standalone:
        print("\n==========================================")
        print("🎤 VOICE ANALYSIS")
        print("==========================================")
        print("This will record your voice ONCE and run:")
        print("  • Speech Emotion Recognition (SER)")
        print("  • Speech-to-Text (Whisper)")
        print("  • Text Emotion Analysis (RoBERTa)")

    if not SEREngine:
        print("❌ SER Engine not available.")
        return None
    if not whisper:
        print("❌ Whisper not available.")
        return None

    # ── Step 1: Record Audio (only if not pre-supplied) ──
    if wav_path is None:
        if not record_audio:
            print("❌ Audio recorder not available.")
            return None

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
            return None

        print(f"\n✅ Audio saved: {wav_path}")

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
    text_emotion_str = "N/A"
    if stt_result and stt_result != "N/A":
        print("\n💬 Analyzing Text Emotion...")
        try:
            te_results = analyze_text_emotion(stt_result, threshold=0.05)
            if te_results:
                text_emotion_str = ", ".join(
                    [f"{r['label']} ({r['score']:.2f})" for r in te_results[:4]]
                )
        except Exception as e:
            print(f"⚠️  Text emotion analysis failed: {e}")

    results = {
        "ser_result": ser_result,
        "stt_result": stt_result,
        "text_emotion_str": text_emotion_str,
    }

    # ── Print report only when called standalone (Option 2) ──
    if is_standalone:
        print("\n" + "="*50)
        print("📊 VOICE ANALYSIS REPORT")
        print("="*50)
        print(f"\n🗣  Speech Emotion:")
        print(f"   → {ser_result}")
        print(f"\n📝 Transcription:")
        print(f"   → \"{stt_result}\"")
        print(f"\n💬 Text Emotion:")
        print(f"   → {text_emotion_str}")
        print("\n" + "="*50 + "\n")

    return results


# ===============================
# OPTION 3 — Multimodal Recording (Face + Voice)
# ===============================

def run_multimodal_recording():
    """
    Records face (OpenFace) and audio simultaneously.
    User presses Enter to stop both recordings.
    Then runs: Face analysis + SER + Whisper + Text Emotion.
    """
    print("\n==========================================")
    print("🎬 MULTIMODAL RECORDING")
    print("==========================================")
    print("This will simultaneously record:")
    print("  🎥 Webcam  → Face Emotion Analysis (OpenFace)")
    print("  🎤 Microphone → SER + STT + Text Emotion")
    print()
    print("⚠️  A webcam window will open for face recording.")
    print("🛑 Press ENTER here to stop ALL recordings when done.")
    print()

    # ── Validate dependencies ──
    if not record_audio:
        print("❌ Audio recorder not available.")
        return

    OPENFACE_EXE = os.path.join(
        SCRIPT_DIR, "external", "openface",
        "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe"
    )
    PROCESSED_DIR = os.path.join(SCRIPT_DIR, "data", "processed")
    DATA_DIR = os.path.join(SCRIPT_DIR, "data", "recordings")

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wav_path = os.path.join(DATA_DIR, f"multimodal_{timestamp}.wav")
    csv_name = f"multimodal_{timestamp}"
    csv_path = os.path.join(PROCESSED_DIR, f"{csv_name}.csv")

    openface_process = None
    audio_exception = [None]   # mutable container for thread error passing

    # ── Audio recording thread ──
    # We record in 60-second chunks; user stops via Enter before that.
    # sounddevice non-blocking pattern: start → user stops → sd.stop()
    try:
        import sounddevice as sd
        import numpy as np
        from scipy.io.wavfile import write as wav_write
    except ImportError as e:
        print(f"❌ sounddevice/scipy not available: {e}")
        return

    FS = 16000
    MAX_DURATION = 120   # safety cap: 2 minutes

    recorded_chunks = []

    def audio_callback(indata, frames, time_info, status):
        recorded_chunks.append(indata.copy())

    # ── Start OpenFace (non-blocking) ──
    if os.path.exists(OPENFACE_EXE):
        try:
            openface_cmd = [
                OPENFACE_EXE,
                "-device", "0",
                "-out_dir", PROCESSED_DIR,
                "-of", csv_name
            ]
            cwd = os.path.dirname(OPENFACE_EXE)
            openface_process = subprocess.Popen(
                openface_cmd, cwd=cwd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"🎥 OpenFace started (PID: {openface_process.pid})")
        except Exception as e:
            print(f"⚠️  OpenFace could not start: {e}")
            openface_process = None
    else:
        print(f"⚠️  OpenFace not found at expected path — skipping face recording.")

    # ── Start audio (non-blocking InputStream) ──
    print("🎤 Microphone open — speak now...")
    print()
    print("  [ Press ENTER to stop recording ]")
    print()

    start_time = time.time()

    try:
        stream = sd.InputStream(
            samplerate=FS,
            channels=1,
            dtype="float32",
            callback=audio_callback
        )
        stream.start()
    except Exception as e:
        print(f"❌ Could not open microphone: {e}")
        if openface_process:
            openface_process.terminate()
        return

    # ── Wait for user to press Enter ──
    try:
        input()   # blocks until Enter
    except (KeyboardInterrupt, EOFError):
        pass

    elapsed = time.time() - start_time

    # ── Stop audio ──
    stream.stop()
    stream.close()
    print(f"\n🔇 Microphone closed. ({elapsed:.1f}s recorded)")

    # ── Stop OpenFace ──
    if openface_process and openface_process.poll() is None:
        openface_process.terminate()
        try:
            openface_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            openface_process.kill()
        print("🎥 OpenFace stopped.")

    # ── Save WAV ──
    if recorded_chunks:
        audio_data = np.concatenate(recorded_chunks, axis=0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_write(wav_path, FS, audio_int16)
        print(f"✅ Audio saved: {wav_path}")
    else:
        print("⚠️  No audio was captured.")
        wav_path = None

    print()
    print("="*50)
    print("🧠 RUNNING ANALYSIS...")
    print("="*50)

    # ── Voice + STT + Text Emotion (reuse Option 2) ──
    voice_results = None
    if wav_path and os.path.exists(wav_path):
        voice_results = run_voice_combined_pipeline(wav_path=wav_path)
    else:
        print("⚠️  Skipping voice analysis — no audio file.")

    # ── Face Emotion Analysis ──
    face_timeline = "N/A"
    if os.path.exists(csv_path):
        try:
            from src.faceexpression.classifier import analyze_openface_csv
            print("\n🙂 Analyzing Face Expressions...")
            time.sleep(1)   # brief buffer for OpenFace file I/O to flush
            face_timeline = analyze_openface_csv(csv_path) or "N/A"
        except Exception as e:
            print(f"⚠️  Face analysis failed: {e}")
    else:
        print("⚠️  Face CSV not found — face analysis skipped.")

    # ── Combined Final Report ──
    ser_result = voice_results["ser_result"] if voice_results else "N/A"
    stt_result = voice_results["stt_result"] if voice_results else "N/A"
    text_emotion_str = voice_results["text_emotion_str"] if voice_results else "N/A"

    print()
    print("="*60)
    print("📊 MULTIMODAL ANALYSIS REPORT")
    print("="*60)
    print(f"\n📝 Transcription:")
    print(f"   → \"{stt_result}\"")
    print(f"\n💬 Text Emotion:")
    print(f"   → {text_emotion_str}")
    print(f"\n🗣  Voice Emotion (SER):")
    print(f"   → {ser_result}")
    print(f"\n🙂 Face Emotion Timeline:")
    if face_timeline and face_timeline != "N/A":
        for line in face_timeline.strip().splitlines():
            print(f"   {line}")
    else:
        print("   → N/A")
    print()
    print("="*60 + "\n")


# ===============================
# MAIN MENU
# ===============================

def main():
    while True:
        print("\n==========================================")
        print("   HUMANOID ASSISTANT - MAIN MENU")
        print("==========================================")
        print("  1. 💬 Text Emotion Analysis  (Keyboard Input)")
        print("  2. 🎤 Voice Analysis          (Speech + Emotion)")
        print("  3. 🎬 Multimodal Recording    (Video + Voice Analysis)")
        print("  4. 🌐 Live Multimodal Chat    (Real-Time Streaming)")
        print("  5. 🚪 Exit")
        print("==========================================")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            run_text_emotion_input()

        elif choice == '2':
            run_voice_combined_pipeline()

        elif choice == '3':
            run_multimodal_recording()

        elif choice == '4':
            print("\n==========================================")
            print("🌐 LIVE MULTIMODAL CHAT")
            print("==========================================")
            print("All three modalities (Face, Voice, Text) will run simultaneously.")
            print("Press Ctrl+C at any time to end the session.\n")
            run_live_streaming_session()

        elif choice == '5':
            print("\n👋 Goodbye!\n")
            break

        else:
            print("\n⚠️  Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()
