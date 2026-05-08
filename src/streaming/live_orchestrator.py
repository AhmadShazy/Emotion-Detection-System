import queue
import time
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

from src.streaming.audio_stream import AudioStreamer
from src.streaming.streaming_stt import StreamingSTT
from src.streaming.streaming_ser import StreamingSER
from src.streaming.streaming_face import StreamingFace
from src.streaming.assistant_response_engine import AssistantResponseEngine
from src.text_emotion.analysis import analyze_text_emotion
from src.streaming.unified_pipeline import build_text_state, process_and_print_unified_json


def _safe_emotion(state: dict, default: str = "neutral") -> str:
    """
    Safely extract emotion string from a state dict.
    Returns default when state is None, key missing, or value is None/empty.

    Fixes Issues 11 & 12: voice_state["emotion"] and face_state["emotion"]
    can be None when workers haven't processed enough audio yet.
    Passing None to _align_emotion() corrupts scores dict with None key.
    """
    if not state:
        return default
    val = state.get("emotion")
    if not val:
        return default
    return str(val)


def run_live_streaming_session():
    print("\n=======================================================")
    print(">>> HUMANOID ASSISTANT V2.1 - TURN-BASED INTERACTION")
    print("=======================================================")

    stt_audio_queue = queue.Queue(maxsize=100)
    ser_audio_queue = queue.Queue(maxsize=100)
    text_stt_queue  = queue.Queue()
    ui_status_queue = queue.Queue()

    print("\n[INIT] Booting components...")

    from src.text_emotion.analysis import load_emotion_model
    load_emotion_model()

    audio_streamer = AudioStreamer()
    audio_streamer.add_queue(stt_audio_queue)
    audio_streamer.add_queue(ser_audio_queue)

    stt_worker = StreamingSTT(
        audio_queue=stt_audio_queue,
        text_queue=text_stt_queue,
        status_queue=ui_status_queue,
        model_size="tiny",
        trailing_silence_seconds=1.5,
    )
    ser_worker = StreamingSER(audio_queue=ser_audio_queue, emotion_queue=None)

    timestamp    = time.strftime("%Y-%m-%d-%H-%M-%S")
    csv_path     = os.path.join(PROJECT_ROOT, "data", "processed",
                                f"live_stream_{timestamp}.csv")
    openface_exe = os.path.join(PROJECT_ROOT, "external", "openface",
                                "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")
    face_worker  = StreamingFace(face_queue=None, csv_path=csv_path,
                                 openface_exe=openface_exe)

    response_engine = AssistantResponseEngine()

    try:
        stt_worker.start()
        ser_worker.start()
        face_worker.start()
        audio_streamer.start()

        print("\n[OK] System Live! Speak and show expressions into the camera.")
        print("Press Ctrl+C to terminate the live session...\n")

        sys.stdout.write("\r[ 💤 Waiting for speech...  ]")
        sys.stdout.flush()

        while True:
            try:
                try:
                    ui_state = ui_status_queue.get_nowait()
                    if ui_state == "LISTENING":
                        sys.stdout.write("\r[ 🎤 Listening to user...   ]")
                    elif ui_state == "ANALYZING":
                        sys.stdout.write("\r[ ⚙️  Analyzing speech...   ]")
                    sys.stdout.flush()
                except queue.Empty:
                    pass

                text        = text_stt_queue.get(timeout=0.1)
                text_state  = build_text_state(
                    text, analyze_text_emotion(text, threshold=0.1)
                )
                voice_state = ser_worker.get_current_emotion()
                face_state  = face_worker.get_current_emotion()

                process_and_print_unified_json(
                    text_state=text_state,
                    voice_state=voice_state,
                    face_state=face_state,
                    raw_text=text,
                    voice_emo_raw=_safe_emotion(voice_state),
                    face_emo_raw=_safe_emotion(face_state),
                )

                ser_worker.clear_buffer()
                face_worker.clear_buffer()

                sys.stdout.write("\n\n\r[ 💤 Waiting for speech...  ]")
                sys.stdout.flush()

            except queue.Empty:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[!] Shutting down streaming system...")
    finally:
        audio_streamer.stop()
        stt_worker.stop()
        ser_worker.stop()
        face_worker.stop()
        stt_worker.join(timeout=2)
        ser_worker.join(timeout=2)
        face_worker.join(timeout=2)
        print("[OK] Shutdown complete.")


if __name__ == "__main__":
    run_live_streaming_session()


# ════════════════════════════════════════════════════════════════════════════
# WebSocket variant — used by routers/stream.py
# ════════════════════════════════════════════════════════════════════════════

def run_live_streaming_session_ws(on_payload, stop_event, session_id=None):
    """
    Runs the live streaming pipeline for a WebSocket session.

    Parameters
    ----------
    on_payload  : callable(dict) — called after each completed speech turn;
                  bridges the result to the async WebSocket send_json().
    stop_event  : threading.Event — set by the WS handler on client disconnect
                  or error; all workers poll this to know when to stop.
    session_id  : str | None — optional; pass a prior session_id to inherit
                  conversation history (LS3 fix). None = new session auto-created.
    """
    print("\n=======================================================")
    print(">>> HUMANOID ASSISTANT V2.1 - LIVE STREAM (WebSocket)")
    print("=======================================================")
    if session_id:
        print(f"[Stream] Inheriting session: {session_id}")

    stt_audio_queue = queue.Queue(maxsize=100)
    ser_audio_queue = queue.Queue(maxsize=100)
    text_stt_queue  = queue.Queue()
    ui_status_queue = queue.Queue()

    print("\n[INIT] Booting components...")

    from src.text_emotion.analysis import load_emotion_model
    load_emotion_model()

    audio_streamer = AudioStreamer()
    audio_streamer.add_queue(stt_audio_queue)
    audio_streamer.add_queue(ser_audio_queue)

    stt_worker = StreamingSTT(
        audio_queue=stt_audio_queue,
        text_queue=text_stt_queue,
        status_queue=ui_status_queue,
        model_size="tiny",
        trailing_silence_seconds=1.5,
    )
    ser_worker = StreamingSER(audio_queue=ser_audio_queue, emotion_queue=None)

    timestamp    = time.strftime("%Y-%m-%d-%H-%M-%S")
    csv_path     = os.path.join(PROJECT_ROOT, "data", "processed",
                                f"ws_stream_{timestamp}.csv")
    openface_exe = os.path.join(PROJECT_ROOT, "external", "openface",
                                "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")
    face_worker  = StreamingFace(face_queue=None, csv_path=csv_path,
                                 openface_exe=openface_exe)

    try:
        stt_worker.start()
        ser_worker.start()
        face_worker.start()
        audio_streamer.start()

        print("\n[OK] WS Stream live — waiting for speech...")

        while not stop_event.is_set():
            try:
                try:
                    ui_state = ui_status_queue.get_nowait()
                    label    = "🎤 Listening" if ui_state == "LISTENING" else "⚙️  Analyzing"
                    print(f"\r[ {label}... ]", end="", flush=True)
                except queue.Empty:
                    pass

                text        = text_stt_queue.get(timeout=0.1)
                text_state  = build_text_state(
                    text, analyze_text_emotion(text, threshold=0.1)
                )
                voice_state = ser_worker.get_current_emotion()
                face_state  = face_worker.get_current_emotion()

                # _safe_emotion() prevents None crashes when workers haven't
                # accumulated enough audio to classify yet.
                # LS3 fix: session_id passed so turns accumulate in one session.
                process_and_print_unified_json(
                    text_state=text_state,
                    voice_state=voice_state,
                    face_state=face_state,
                    raw_text=text,
                    voice_emo_raw=_safe_emotion(voice_state),
                    face_emo_raw=_safe_emotion(face_state),
                    session_id=session_id,
                    on_payload=on_payload,
                )

                ser_worker.clear_buffer()
                face_worker.clear_buffer()

                print("\r[ 💤 Waiting for speech... ]", end="", flush=True)

            except queue.Empty:
                time.sleep(0.01)

    finally:
        audio_streamer.stop()
        stt_worker.stop()
        ser_worker.stop()
        face_worker.stop()
        stt_worker.join(timeout=2)
        ser_worker.join(timeout=2)
        face_worker.join(timeout=2)
        print("\n[OK] WS stream shutdown complete.")