import sys
import os

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is in python path
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Import SER main
try:
    from src.ser.record_analyze import main as ser_main
except ImportError as e:
    print(f"Warning: Could not import SER module: {e}")
    ser_main = None

try:
    from src.faceexpression.record_express import run_face_pipeline
except ImportError as e:
    print(f"Warning: Could not import Face Expression module: {e}")
    # Define a dummy function if import fails
    def run_face_pipeline():
        print("❌ Face Expression module not available.")

try:
    from src.stt.record_transcribe import stt_pipeline
except ImportError as e:
    print(f"Warning: Could not import STT module: {e}")
    def stt_pipeline():
        print("❌ STT module not available.")

try:
    from src.full_analysis.main import run_full_analysis
except ImportError as e:
    print(f"Warning: Could not import Full Analysis module: {e}")
    def run_full_analysis():
        print("❌ Full Analysis module not available.")



def run_ser_pipeline_wrapper():
    if ser_main:
        print("\n==========================================")
        print("🎤 STARTING VOICE EMOTION PIPELINE")
        print("==========================================")
        ser_main()
    else:
        print("❌ SER module not available.")

def main():
    while True:
        print("\n==========================================")
        print("🤖 HUMANOID ASSISTANT - MAIN MENU")
        print("==========================================")
        print("1. Face Expression Analysis (OpenFace)")
        print("2. Voice Emotion Analysis (SER_SpeechBrain)")
        print("3. Speech-to-Text (STT_Whisper)")
        print("4. Full Analysis (Face + Voice + STT)")

        print("5. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            run_face_pipeline()
        elif choice == '2':
            run_ser_pipeline_wrapper()
        elif choice == '3':
            stt_pipeline()
        elif choice == '4':
            run_full_analysis()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
