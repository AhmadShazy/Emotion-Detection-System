import time
import os
try:
    from .recorder import record_audio
    from .ser_engine import SEREngine
except ImportError:
    # Fallback for running directly
    from recorder import record_audio
    from ser_engine import SEREngine

# Define project root and data directory
# Assuming this script is at src/ser/main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def main():
    print("Initializing Speech Emotion Recognition System...", flush=True)
    try:
        engine = SEREngine()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have installed dependencies correctly.")
        print("Try: pip install speechbrain transformers torchaudio torch --index-url https://download.pytorch.org/whl/cpu")
        return

    while True:
        choice = input("\nPress 'Ent' to record, or 'q' to quit: ")
        if choice.lower() == 'q':
            break

        print("\n--- Starting Session ---")
        try:
            # 1. Record
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            wav_file = os.path.join(DATA_DIR, f"ser_{timestamp}.wav")
            print(f"Recording to {wav_file}...")
            record_audio(duration=10, filename=wav_file) # Reduced to 5s for quick testing, user can change later
            
            # 2. Analyze
            print("Analyzing emotion...")
            emotion = engine.predict_emotion(wav_file)
            
            # 3. Result
            print("\n" + "="*30)
            print(f"DETECTED EMOTION: {emotion}")
            print("="*30 + "\n")
            
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
