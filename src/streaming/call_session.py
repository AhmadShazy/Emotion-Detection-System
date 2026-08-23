"""
src/streaming/call_session.py
=============================
Everything one live caller owns.

The old streaming design ran a set of worker THREADS per session, fed by the
server's own microphone through sounddevice. That could only ever serve the one
person sitting at the machine, and the threads communicated with the event loop
through `asyncio.run_coroutine_threadsafe`, which blocked a worker on the loop
and treated a momentarily-busy loop as a dead client.

Here, media arrives over the WebSocket, so the event loop is already the
producer and there is nothing to bridge. A CallSession owns no threads at all:
it is plain state that the connection's coroutine mutates, plus one blocking
`analyze()` that gets handed to a shared executor.

The expensive models stay shared and global (ModelRegistry) — RoBERTa alone is
~350MB and takes 30s+ to load, so per-connection instances are out of the
question.
"""

import time
import numpy as np

from src.streaming.turn_detector import TurnDetector, SAMPLE_RATE

# How much audio a connection may hold. Sized to comfortably exceed the longest
# permitted turn, so a normal caller never touches the limit.
RING_SECONDS = 30
RING_SAMPLES = RING_SECONDS * SAMPLE_RATE

# Frames kept for the face reading. At the browser's ~3fps this is ~30s of
# video, matching the audio ring.
MAX_VIDEO_FRAMES = 90

# Above this, a frame is refused rather than buffered. A caller sending 4K
# stills would otherwise consume memory for no analytical benefit — MediaPipe
# resizes to fixed model inputs internally, so resolution beyond ~480p buys
# nothing.
MAX_FRAME_BYTES = 512 * 1024


class CallSession:
    """
    One live call. Not thread-safe by design — only the connection's own
    coroutine touches it, except `analyze()` which runs on the executor and
    reads a snapshot passed to it explicitly.
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id
        self.detector = TurnDetector()

        # Video is kept as raw JPEG BYTES, not decoded arrays. Decoding 90
        # frames up front would cost ~1.5s of CPU for frames a turn may never
        # use; decoding lazily means only the frames a turn actually covers are
        # ever touched.
        self.video_frames: list = []      # [(timestamp, jpeg_bytes)]

        self.created_at = time.time()
        self.audio_samples_received = 0
        self.video_frames_received = 0
        self.turns_emitted = 0
        self.turns_shed = 0
        self.closed = False

        # At most one turn in flight. A second turn completing while the first
        # is still running is shed rather than queued — an unbounded backlog is
        # exactly the failure this guards against, and stale emotion is worth
        # little anyway.
        self.analysis_in_flight = False

    # ── Ingest ───────────────────────────────────────────────────────────────

    def push_audio(self, pcm_bytes: bytes):
        """
        Accepts one binary audio frame and returns a completed turn, or None.

        Wire format is signed 16-bit little-endian at 16 kHz mono. Converting
        to float32 here is a dtype change at a fixed rate, NOT resampling — the
        browser is responsible for producing 16 kHz, because nothing in this
        codebase resamples and SpeechBrain silently returns a confident WRONG
        label if handed the wrong rate.
        """
        if not pcm_bytes:
            return None

        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        self.audio_samples_received += len(samples)

        turn = self.detector.push(samples)
        if turn is None:
            return None

        # Attach the video that overlaps this turn, then drop older frames.
        turn["video"] = self._take_video_for_turn(turn["duration"])
        return turn

    def push_video(self, jpeg_bytes: bytes) -> bool:
        """
        Accepts one JPEG still. Returns False if refused.

        Refusing an oversized frame is deliberate: it protects memory without
        killing the call, and the caller simply loses one frame out of the
        dozens that make up a turn's majority vote.
        """
        if not jpeg_bytes or len(jpeg_bytes) > MAX_FRAME_BYTES:
            return False

        self.video_frames.append((time.time(), jpeg_bytes))
        self.video_frames_received += 1

        # Bound the buffer. Oldest frames go first — for a live call the recent
        # face is the relevant one.
        if len(self.video_frames) > MAX_VIDEO_FRAMES:
            del self.video_frames[:len(self.video_frames) - MAX_VIDEO_FRAMES]

        return True

    def _take_video_for_turn(self, duration: float) -> list:
        """
        Returns the JPEG frames captured while this turn was being spoken.

        Uses wall-clock rather than sample counts because the two streams
        arrive independently and are not sample-synchronised. A small margin is
        added so a frame captured just as speech began is not missed.
        """
        cutoff = time.time() - (duration + 1.0)
        frames = [jpg for ts, jpg in self.video_frames if ts >= cutoff]

        # Everything older than this turn is now useless.
        self.video_frames = [
            (ts, jpg) for ts, jpg in self.video_frames if ts >= cutoff
        ]

        return frames

    # ── Overflow ─────────────────────────────────────────────────────────────

    def check_overflow(self) -> bool:
        """
        True when an in-progress turn has grown past what we will hold.

        The caller should abandon the turn and tell the user. Splicing around a
        gap would be worse: Whisper does not error on a discontinuity, it
        transcribes the seam into a plausible-looking wrong sentence.
        """
        return self.detector.turn_samples >= RING_SAMPLES

    def stats(self) -> dict:
        return {
            "session_id":      self.session_id,
            "uptime_seconds":  round(time.time() - self.created_at, 1),
            "audio_seconds":   round(self.audio_samples_received / SAMPLE_RATE, 1),
            "video_frames":    self.video_frames_received,
            "turns_emitted":   self.turns_emitted,
            "turns_shed":      self.turns_shed,
            "detector":        self.detector.stats(),
        }


# ── Analysis, run on the shared executor ──────────────────────────────────────

def analyze_turn(audio: np.ndarray, jpeg_frames: list, session_id: str | None) -> dict:
    """
    Runs one completed turn through the full pipeline and returns the payload.

    BLOCKING and CPU-bound — must be called via an executor, never on the event
    loop. Measured on this machine for a 5s turn: STT ~2.7s, SER ~1.7s, face
    ~0.9s, so roughly 4-5s wall clock.

    Deliberately reuses the same functions the file-upload path uses, so live
    and uploaded video cannot drift apart in behaviour.
    """
    from src.interactive_modes import _is_hallucination
    from src.streaming.unified_pipeline import build_text_state, process_and_print_unified_json
    from src.text_emotion.analysis import analyze_text_emotion
    from src.faceexpression.mediapipe_analyzer import analyze_jpeg_frames
    from src.core.model_registry import registry

    # ── Face ─────────────────────────────────────────────────────────────────
    face_state = None
    try:
        face_state = analyze_jpeg_frames(jpeg_frames)
    except Exception as exc:
        print(f"[Live] Face analysis failed: {exc}")

    # ── Speech to text ───────────────────────────────────────────────────────
    transcript = ""
    try:
        model = registry.get("faster_whisper")
        segments, _ = model.transcribe(audio, beam_size=1)
        raw = " ".join(
            s.text for s in segments if getattr(s, "no_speech_prob", 0.0) < 0.60
        ).strip()
        if raw and not _is_hallucination(raw):
            transcript = raw
    except Exception as exc:
        print(f"[Live] STT failed: {exc}")

    # ── Voice emotion ────────────────────────────────────────────────────────
    voice_state = None
    ser_label = "neutral"
    try:
        import torch
        classifier = registry.get("speechbrain")
        _, score, _, text_lab = classifier.classify_batch(
            torch.from_numpy(audio).unsqueeze(0)
        )
        label_map = {"hap": "Happy", "ang": "Angry", "neu": "Neutral", "sad": "Sad"}
        ser_label = label_map.get(text_lab[0], text_lab[0])
        confidence = float(score[0])
        voice_state = {
            "source":          "voice",
            "emotion":         ser_label,
            "confidence":      round(confidence, 4),
            "average_emotion": ser_label,
            "peak_emotion":    ser_label,
            "reliability":     round(min(1.0, confidence + 0.15), 4),
        }
    except Exception as exc:
        print(f"[Live] SER failed: {exc}")

    # ── Text emotion ─────────────────────────────────────────────────────────
    text_state = None
    if transcript:
        try:
            text_state = build_text_state(
                transcript, analyze_text_emotion(transcript, threshold=0.05)
            )
        except Exception as exc:
            print(f"[Live] Text emotion failed: {exc}")

    # ── Fuse and build the payload ───────────────────────────────────────────
    return process_and_print_unified_json(
        text_state=text_state,
        voice_state=voice_state,
        face_state=face_state,
        raw_text=transcript,
        voice_emo_raw=ser_label,
        face_emo_raw=face_state["emotion"] if face_state else "neutral",
        session_id=session_id,
    )
