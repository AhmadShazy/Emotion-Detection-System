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

# Frames kept for the face reading. At the browser's ~3fps this is ~30s of
# video, which comfortably covers the longest permitted turn.
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

# Cap on how much audio the voice-emotion model sees.
#
# SER cost is linear in input length — measured ~250ms per second of audio, so a
# 20-second turn spends 5.2s in SER alone. Emotion is also weighted toward how
# someone finishes a thought rather than how they began it, so the tail is the
# informative part. The original streaming implementation capped at 5s for the
# same reason ("prevent CPU hang on long unbroken sentences").
SER_MAX_SECONDS = 6.0


def analyze_turn(audio: np.ndarray, jpeg_frames: list, session_id: str | None) -> dict:
    """
    Runs one completed turn through the full pipeline and returns the payload.

    BLOCKING and CPU-bound — must be called via an executor, never on the event
    loop.

    The three analyses are independent, so they run CONCURRENTLY. They are
    C-level calls that release the GIL, and measured on this machine that saves
    real time even though they contend for the same cores. Text emotion runs
    afterwards because it needs the transcript.

    Deliberately reuses the same functions the file-upload path uses, so live
    and uploaded video cannot drift apart in behaviour.
    """
    import concurrent.futures

    from src.interactive_modes import _is_hallucination
    from src.streaming.unified_pipeline import (
        build_text_state, build_voice_state, process_and_print_unified_json,
    )
    from src.text_emotion.analysis import analyze_text_emotion
    from src.faceexpression.mediapipe_analyzer import analyze_jpeg_frames
    from src.core.model_registry import registry

    # ── The three independent stages ─────────────────────────────────────────

    def run_face():
        try:
            return analyze_jpeg_frames(jpeg_frames)
        except Exception as exc:
            print(f"[Live] Face analysis failed: {exc}")
            return None

    def run_stt():
        try:
            model = registry.get("faster_whisper")
            segments, _ = model.transcribe(audio, beam_size=1)
            raw = " ".join(
                s.text for s in segments if getattr(s, "no_speech_prob", 0.0) < 0.60
            ).strip()
            return raw if raw and not _is_hallucination(raw) else ""
        except Exception as exc:
            print(f"[Live] STT failed: {exc}")
            return ""

    def run_ser():
        try:
            import torch
            classifier = registry.get("speechbrain")

            # Only the tail, so cost stays flat regardless of how long the
            # caller talked.
            limit = int(SER_MAX_SECONDS * SAMPLE_RATE)
            clip = audio[-limit:] if len(audio) > limit else audio

            _, score, _, text_lab = classifier.classify_batch(
                torch.from_numpy(clip).unsqueeze(0)
            )
            label_map = {"hap": "Happy", "ang": "Angry", "neu": "Neutral", "sad": "Sad"}
            label = label_map.get(text_lab[0], text_lab[0])
            confidence = float(score[0])
            # Shared builder, so this path and the upload paths cannot disagree
            # about how confidence becomes reliability.
            return label, build_voice_state(label, confidence)
        except Exception as exc:
            print(f"[Live] SER failed: {exc}")
            return "neutral", None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_face = pool.submit(run_face)
        f_stt  = pool.submit(run_stt)
        f_ser  = pool.submit(run_ser)

        face_state = f_face.result()
        transcript = f_stt.result()
        ser_label, voice_state = f_ser.result()

    # ── Text emotion — needs the transcript, so it cannot start earlier ──────
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
