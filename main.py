import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from src.interactive_modes import (
    run_text_emotion_input,
    run_voice_combined_pipeline,
    run_multimodal_recording
)

try:
    from src.streaming.live_orchestrator import run_live_streaming_session
except ImportError as e:
    print(f"Warning: Could not import V2 Streaming module: {e}")
    def run_live_streaming_session():
        print("❌ V2 Streaming module not available.")

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
