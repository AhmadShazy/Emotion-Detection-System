import os
import time
import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import sys
# Ensure project root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.text_emotion.analysis import analyze_text_emotion
except ImportError:
    print("⚠️ Could not import text emotion module.")
    def analyze_text_emotion(text): return None

# ----------------------------
# CONFIGURATION
# ----------------------------

def stt_pipeline():
    print("\n==========================================")
    print("📝 STARTING SPEECH-TO-TEXT PIPELINE")
    print("==========================================")
    # ----------------------------
    # CONFIGURATION
    # ----------------------------
    fs = 16000                # Sample rate
    total_seconds = 10        # Total listening time
    countdown_seconds = 3     # Countdown before mic closes

    # ----------------------------
    # GENERATE TIMESTAMPED FILENAME
    # ----------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Ensure we use absolute paths relative to this script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "recordings")

    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    audio_file = os.path.join(AUDIO_DIR, f"mic_{timestamp}.wav")


    # ----------------------------
    # STEP 1: START RECORDING
    # ----------------------------
    print("🎤 Mic is OPEN")
    print("👉 Speak now...")

    audio = sd.rec(
        int(total_seconds * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )

    # ----------------------------
    # STEP 2: COUNTDOWN
    # ----------------------------
    time.sleep(total_seconds - countdown_seconds)

    for i in range(countdown_seconds, 0, -1):
        print(f"⏳ Mic closing in {i}...")
        time.sleep(1)

    sd.wait()  # Ensure recording is complete

    write(audio_file, fs, audio)

    print("🔇 Mic CLOSED")
    print(f"✅ Audio saved as: {audio_file}")

    # ----------------------------
    # STEP 3: WAIT BEFORE TRANSCRIPTION
    # ----------------------------
    time.sleep(2)

    # ----------------------------
    # STEP 4: TRANSCRIBE
    # ----------------------------
    print("🧠 Loading Whisper model...")
    model_dir = os.path.join(SCRIPT_DIR, "..", "..", "external", "whisper")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = whisper.load_model("base", download_root=model_dir)

    print("📖 Transcribing audio...")
    result = model.transcribe(audio_file)

    # ----------------------------
    # STEP 5: OUTPUT REPORT
    # ----------------------------
    
    # Analyze emotion first (so logs appear before the final report)
    print("\n🧠 Analyzing Text Emotion...")
    emotion_results = analyze_text_emotion(result["text"])
    
    emotion_str = "Could not determine"
    if emotion_results:
        # Format as "Joy (0.90), Excitement (0.45)"
        emotion_str = ", ".join([f"{res['label']} ({res['score']:.2f})" for res in emotion_results])

    print("\n" + "="*50)
    print("📊 STT ANALYSIS REPORT")
    print("="*50)
    print(f"💭 Text Emotion  : {emotion_str}")
    print(f"📝 Transcription : \"{result['text'].strip()}\"")
    print("="*50 + "\n")

if __name__ == "__main__":
    stt_pipeline()
