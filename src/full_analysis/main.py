import os
import time
import threading
import subprocess
import whisper
import sys

# Ensure project root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ser.recorder import record_audio
from src.ser.ser_engine import SEREngine
from src.faceexpression.classifier import analyze_openface_csv
from src.text_emotion.analysis import analyze_text_emotion

# CONFIG
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OPENFACE_DIR = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64")
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
WHISPER_MODEL_DIR = os.path.join(PROJECT_ROOT, "external", "whisper")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)
if not os.path.exists(WHISPER_MODEL_DIR):
    os.makedirs(WHISPER_MODEL_DIR)

def run_full_analysis():
    print("\n==========================================")
    print("🚀 STARTING FULL ANALYSIS (20s Session)")
    print("==========================================")
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    wav_filename = f"full_analysis_{timestamp}.wav"
    wav_path = os.path.join(DATA_DIR, wav_filename)
    
    csv_filename = f"full_analysis_{timestamp}"
    csv_path = os.path.join(PROCESSED_DIR, f"{csv_filename}.csv")

    # ---------------------------------------------------------
    # 1. PARALLEL RECORDING
    # ---------------------------------------------------------
    print("\n[1/3] 🎥🎤 Preparing Parallel Recording...")
    
    # Audio Thread
    def record_audio_task():
        # record_audio prints its own messages
        record_audio(duration=20, filename=wav_path)

    audio_thread = threading.Thread(target=record_audio_task)
    
    # OpenFace Command
    openface_cmd = [
        OPENFACE_EXE,
        "-device", "0",
        "-out_dir", PROCESSED_DIR,
        "-of", csv_filename
    ]
    
    print("👉 Starting Recording NOW (Duration: 20s)...")
    
    # Start both
    audio_thread.start()
    
    try:
        # Start OpenFace (Process)
        # We communicate via stdin/stdout if needed, but mostly we just let it run
        # We redirect stdout/stderr to suppress some noise, or keep it if user wants to see
        # Let's suppress OpenFace output to keep console clean, or maybe let it flow?
        # User requested "seamless", let's suppress OpenFace log spam if possible, but OpenFace prints useful info.
        # Let's keep it visible but maybe indented? No, too complex. Just run it.
        openface_process = subprocess.Popen(openface_cmd, cwd=OPENFACE_DIR)
        
        # Wait for Audio to finish (since it has the explicit 20s timer)
        audio_thread.join()
        
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        if openface_process:
            openface_process.kill()
        return

    print("✅ Audio Recording Complete.")
    print("⏳ Stopping Video Recording...")
    
    # Terminate OpenFace
    if openface_process.poll() is None:
        openface_process.terminate()
        try:
            openface_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            openface_process.kill()
            
    print("✅ Video Recording Complete.")

    # ---------------------------------------------------------
    # 2. ANALYSIS
    # ---------------------------------------------------------
    print("\n[2/3] 🧠 Running Analysis...")
    
    # A. SER Analysis
    print("\n--- 🗣️ Voice Emotion Analysis ---")
    ser_result = "N/A"
    try:
        ser_engine = SEREngine() # Loads model
        ser_result = ser_engine.predict_emotion(wav_path)
        print(f"Detected Voice Emotion: {ser_result}")
    except Exception as e:
        print(f"SER Failed: {e}")

    # B. STT Analysis
    print("\n--- 📝 Speech-to-Text (Whisper) ---")
    stt_result = "N/A"
    try:
        model = whisper.load_model("base", download_root=WHISPER_MODEL_DIR)
        transcription = model.transcribe(wav_path)
        stt_result = transcription["text"].strip()
        print(f"Transcription: \"{stt_result}\"")
    except Exception as e:
        print(f"STT Failed: {e}")

    # B.2 Text Emotion Analysis
    text_emotion_result = "N/A"
    if stt_result != "N/A":
        te_res_list = analyze_text_emotion(stt_result)
        if te_res_list:
            text_emotion_result = ", ".join([f"{res['label']} ({res['score']:.2f})" for res in te_res_list])
            print(f"Detected Text Emotion: {text_emotion_result}")

    # C. Face Analysis
    print("\n--- ☺️ Face Expression Analysis ---")
    face_timeline = ""
    try:
        # We need to give a moment for the CSV to be fully written/closed by OpenFace
        time.sleep(1) 
        if os.path.exists(csv_path):
            face_timeline = analyze_openface_csv(csv_path)
            print("Face Analysis Complete.")
        else:
            print("❌ Face CSV not found.")
    except Exception as e:
        print(f"Face Analysis Failed: {e}")

    # ---------------------------------------------------------
    # 3. FINAL SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 FULL ANALYSIS REPORT")
    print("="*50)
    print(f"🗣️ Voice Emotion : {ser_result}")
    print(f"💭 Text Emotion   : {text_emotion_result}")
    print(f"📝 Transcription  : \"{stt_result}\"")
    print("="*50)
    
    if face_timeline:
        print("\n☺️ Face Expression Timeline:")
        print(face_timeline)
        print("="*50 + "\n")
    else:
        print("\n❌ No Face Expression Data available.\n")

if __name__ == "__main__":
    run_full_analysis()
