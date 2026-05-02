import sys
import os
import time
import datetime
import subprocess
import threading
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from src.interactive_modes import (
    process_text_emotion,
    record_audio_clip,
    process_voice_pipeline,
    process_multimodal_data
)
from src.streaming.unified_pipeline import process_and_print_unified_json

try:
    from src.streaming.live_orchestrator import run_live_streaming_session
except ImportError as e:
    print(f"Warning: Could not import V2 Streaming module: {e}")
    def run_live_streaming_session():
        print("❌ V2 Streaming module not available.")

try:
    from src.text_emotion.analysis import load_emotion_model
except ImportError:
    def load_emotion_model(): pass

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError:
    sd = None
    wav_write = None

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
        print("  3. 🎥 Multimodal Recording      (Video + Voice)")
        print("  4. 🌐 Live Multimodal Chat    (Real-Time Streaming)")
        print("  5. 🚪 Exit")
        print("==========================================")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            print("\n==========================================")
            print("💬 TEXT EMOTION ANALYSIS")
            print("==========================================")
            print("Type your text below and press Enter.")
            print("(Type 'quit' to return to main menu)\n")

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
                text_state = process_text_emotion(text)
                
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

        elif choice == '2':
            print("\n==========================================")
            print("🎤 VOICE ANALYSIS")
            print("==========================================")
            print("This will record your voice ONCE and run:")
            print("  • Speech Emotion Recognition (SER)")
            print("  • Speech-to-Text (Whisper)")
            print("  • Text Emotion Analysis (RoBERTa)")
            
            print("\n🎙️  Recording for 10 seconds...")
            print("👉  Speak now!\n")
            
            wav_path = record_audio_clip(duration=10)
            if not wav_path:
                print("❌ Recording failed or audio recorder not available.")
                continue
                
            print(f"\n✅ Audio saved: {wav_path}")
            print("-" * 50)
            print("\n🧠 Processing audio (SER, Whisper, Text Emotion)...")
            
            text_state, voice_state, stt_result, ser_result = process_voice_pipeline(wav_path)
            
            process_and_print_unified_json(
                text_state=text_state,
                voice_state=voice_state,
                face_state=None,
                raw_text=stt_result if stt_result != "N/A" else "",
                voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
                face_emo_raw="neutral"
            )

        elif choice == '3':
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
                continue
                
            OPENFACE_DIR = os.path.join(SCRIPT_DIR, "external", "openface", "OpenFace_2.2.0_win_x64")
            OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
            OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "data", "processed")
            DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "recordings")

            for d in (OUTPUT_DIR, DATA_DIR):
                if not os.path.exists(d):
                    os.makedirs(d)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            of_filename = f"multimodal_{timestamp}"
            csv_path = os.path.join(OUTPUT_DIR, f"{of_filename}.csv")
            wav_path = os.path.join(DATA_DIR, f"multimodal_{timestamp}.wav")

            face_available = os.path.exists(OPENFACE_EXE)
            if not face_available:
                print(f"❌ OpenFace executable not found at: {OPENFACE_EXE}")
                print("   Skipping face recording. Only audio will be analysed.")

            FS = 16000
            audio_chunks = []
            stop_event = threading.Event()

            def _audio_worker():
                with sd.InputStream(samplerate=FS, channels=1, dtype='int16') as stream:
                    while not stop_event.is_set():
                        chunk, _ = stream.read(FS // 2)
                        audio_chunks.append(chunk.copy())

            of_process = None
            if face_available:
                of_cmd = [
                    OPENFACE_EXE,
                    "-device", "0",
                    "-out_dir", OUTPUT_DIR,
                    "-of", of_filename
                ]
                try:
                    of_process = subprocess.Popen(of_cmd, cwd=OPENFACE_DIR,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL)
                    print("\n▶  OpenFace launched (non-blocking).")
                except FileNotFoundError:
                    print("⚠️  Could not launch OpenFace. Face analysis will be skipped.")
                    face_available = False

            audio_thread = threading.Thread(target=_audio_worker, daemon=True)
            audio_thread.start()

            print("\n🔴 RECORDING — press Enter to stop...")
            start_time = time.time()
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                pass
            elapsed = time.time() - start_time
            print(f"\n⏱  Recorded {elapsed:.1f} s.")

            stop_event.set()
            audio_thread.join(timeout=3)

            if of_process is not None and of_process.poll() is None:
                of_process.terminate()
                try:
                    of_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    of_process.kill()
                print("✅ OpenFace stopped.")

            if audio_chunks:
                audio_data = np.concatenate(audio_chunks, axis=0)
                wav_write(wav_path, FS, audio_data)
                print(f"✅ Audio saved: {wav_path}")
            else:
                print("⚠️  No audio captured.")
                wav_path = None

            print("\n" + "="*60)
            print("⚙️  POST-PROCESSING — please wait...")
            print("="*60)
            
            text_state, voice_state, face_state, stt_result, ser_result, face_timeline = process_multimodal_data(
                wav_path, csv_path, face_available
            )
            
            face_emo_raw = face_state["emotion"] if face_state else "neutral"
            
            process_and_print_unified_json(
                text_state=text_state,
                voice_state=voice_state,
                face_state=face_state,
                raw_text=stt_result if stt_result != "N/A" else "",
                voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
                face_emo_raw=face_emo_raw
            )

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
