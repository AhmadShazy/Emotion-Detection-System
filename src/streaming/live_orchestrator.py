import queue
import time
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from src.streaming.audio_stream import AudioStreamer
from src.streaming.streaming_stt import StreamingSTT
from src.streaming.streaming_ser import StreamingSER
from src.streaming.streaming_face import StreamingFace
from src.streaming.emotion_state_manager import EmotionStateManager
from src.streaming.assistant_response_engine import AssistantResponseEngine
from src.text_emotion.analysis import analyze_text_emotion

def run_live_streaming_session():
    print("\n=======================================================")
    print(">>> HUMANOID ASSISTANT V2.1 - TURN-BASED INTERACTION")
    print("=======================================================")
    
    # 1. Initialize Communication Queues
    stt_audio_queue = queue.Queue(maxsize=100)
    ser_audio_queue = queue.Queue(maxsize=100)
    text_stt_queue = queue.Queue()      # STT outputs raw text here
    ui_status_queue = queue.Queue()     # STT VAD sends LISTENING/ANALYZING flags here

    print("\n[INIT] Booting components...")
    
    # Preload the Text Emotion Model (RoBERTa) into memory
    from src.text_emotion.analysis import load_emotion_model
    load_emotion_model()

    # 2. Initialize Workers
    # Audio Input (Non-blocking Broadcaster)
    audio_streamer = AudioStreamer()
    audio_streamer.add_queue(stt_audio_queue)
    audio_streamer.add_queue(ser_audio_queue)

    # STT (Faster-Whisper CPU) waits for trailing silence to extract sentences naturally
    stt_worker = StreamingSTT(audio_queue=stt_audio_queue, text_queue=text_stt_queue, status_queue=ui_status_queue, model_size="tiny", trailing_silence_seconds=1.5)
    
    # SER (Wav2Vec2 Dynamic Build)
    ser_worker = StreamingSER(audio_queue=ser_audio_queue, emotion_queue=None) # queue no longer needed
    
    # Face (OpenFace Subprocess tailing)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", f"live_stream_{timestamp}.csv")
    openface_exe = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")
    face_worker = StreamingFace(face_queue=None, csv_path=csv_path, openface_exe=openface_exe)

    # Synchronous Fusion Brain & Assistant Engine
    state_manager = EmotionStateManager()
    response_engine = AssistantResponseEngine()

    try:
        # Start background sensor workers
        stt_worker.start()
        ser_worker.start()
        face_worker.start()
        
        # Start continuous audio flow
        audio_streamer.start()
        
        print("\n[OK] System Live! Speak and show expressions into the camera.")
        print("Press Ctrl+C to terminate the live session...\n")
        
        # Initial UI State
        sys.stdout.write("\r[ 💤 Waiting for speech...  ]")
        sys.stdout.flush()
        
        # Main Thread Loop acts as the Turn-Based Orchestrator
        while True:
            try:
                # 0. Check for UI State changes from the STT worker (Non-blocking)
                try:
                    ui_state = ui_status_queue.get_nowait()
                    if ui_state == "LISTENING":
                        sys.stdout.write("\r[ 🎤 Listening to user...   ]")
                    elif ui_state == "ANALYZING":
                        sys.stdout.write("\r[ ⚙️ Analyzing speech...    ]")
                    sys.stdout.flush()
                except queue.Empty:
                    pass
                
                # 1. Wait for user to stop speaking & STT to yield a transcribed sentence
                text = text_stt_queue.get(timeout=0.1)
                
                # 2. Run mock Text Emotion Analysis
                text_emotions = analyze_text_emotion(text, threshold=0.1)
                text_emo_str = "Neutral (0.00)"
                text_state = {"emotion": "Neutral", "confidence": 0.0}
                
                if text_emotions and len(text_emotions) > 0:
                    top_emotion = text_emotions[0]['label']
                    top_score = text_emotions[0]['score']
                    text_emo_str = f"{top_emotion} ({top_score:.2f})"
                    text_state = {"emotion": top_emotion, "confidence": top_score}
                    
                # 3. Snapshot the latest SER and Face states
                voice_state = ser_worker.get_current_emotion()
                face_state = face_worker.get_current_emotion()
                
                v_emo = f"{voice_state['emotion']} ({voice_state['confidence']:.2f})" if voice_state['emotion'] else "N/A"
                f_emo = f"{face_state['emotion']} ({face_state['confidence']:.2f})" if face_state['emotion'] else "N/A"
                
                # 4. Fuse the three snapshot modalities synchronously
                unified_emotion = state_manager.fuse(text_state, voice_state, face_state)
                
                # 5. Output cleanly to console
                print("\n" + "-"*60)
                print(f"[💬 STT] \"{text}\"")
                print(f"         ↳ [Text Analysis] Emotion: {text_emo_str}")
                print(f"         ↳ [Voice (SER)  ] Emotion: {v_emo}")
                print(f"         ↳ [Face (Vis)   ] Emotion: {f_emo}")
                print("-" * 60)
                
                # Let the assistant react synchronously right after the sentence block
                response_engine.react(unified_emotion)
                
                # 6. Flush the audio/face buffers so the next sentence isn't polluted with old history
                ser_worker.clear_buffer()
                face_worker.clear_buffer()
                
                # Reset UI state
                sys.stdout.write("\n\n\r[ 💤 Waiting for speech...  ]")
                sys.stdout.flush()
                
            except queue.Empty:
                # Sleep briefly and keep waiting for STT to produce a sentence
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[!] Shutting down streaming system...")
    finally:
        audio_streamer.stop()
        stt_worker.stop()
        ser_worker.stop()
        face_worker.stop()
        
        # Wait for threads
        stt_worker.join(timeout=2)
        ser_worker.join(timeout=2)
        face_worker.join(timeout=2)
        print("[OK] Shutdown complete.")

if __name__ == "__main__":
    run_live_streaming_session()
