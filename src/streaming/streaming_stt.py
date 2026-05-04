"""
src/streaming/streaming_stt.py
================================
Streaming STT using faster-whisper.
Model is obtained from ModelRegistry — never loaded here.
"""

import threading
import queue
import time
import numpy as np

# faster_whisper import kept for type hint clarity but NOT for instantiation
from faster_whisper import WhisperModel


class StreamingSTT(threading.Thread):
    def __init__(
        self,
        audio_queue,
        text_queue,
        status_queue=None,
        model_size="tiny",           # kept for API compatibility, ignored now
        compute_type="int8",         # kept for API compatibility, ignored now
        silence_threshold=0.01,
        trailing_silence_seconds=2.0,
        sample_rate=16000,
    ):
        super().__init__(daemon=True)
        self.audio_queue    = audio_queue
        self.text_queue     = text_queue
        self.status_queue   = status_queue

        self.silence_threshold       = silence_threshold
        self.trailing_silence_frames = int(trailing_silence_seconds * sample_rate)
        self.current_silence_frames  = 0
        self.sample_rate             = sample_rate

        self.running      = False
        self.audio_buffer = np.array([], dtype=np.float32)

        # Get already-loaded model from registry — no download, no I/O
        from src.core.model_registry import registry
        self.model = registry.get("faster_whisper")
        print("[StreamingSTT] Model obtained from registry.")

    # ── Everything below is UNCHANGED from your original ─────────────────────

    def run(self):
        self.running = True
        print("[STT] Streaming STT worker started.")
        has_spoken = False

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

                is_silent = self._is_silent(chunk)

                if is_silent:
                    self.current_silence_frames += len(chunk)
                else:
                    self.current_silence_frames = 0
                    if not has_spoken:
                        has_spoken = True
                        if self.status_queue:
                            self.status_queue.put("LISTENING")

                if self.current_silence_frames >= self.trailing_silence_frames:
                    if has_spoken:
                        if self.status_queue:
                            self.status_queue.put("ANALYZING")
                        self._transcribe_buffer()
                        has_spoken = False
                    else:
                        self.audio_buffer = np.array([], dtype=np.float32)
                    self.current_silence_frames = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[STT] Worker Error: {e}")

    def _is_silent(self, chunk):
        rms = np.sqrt(np.mean(chunk**2))
        return rms < self.silence_threshold

    def _transcribe_buffer(self):
        if len(self.audio_buffer) < self.sample_rate:
            self.audio_buffer = np.array([], dtype=np.float32)
            return

        audio_data        = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.float32)

        try:
            segments, info = self.model.transcribe(audio_data, beam_size=1)
            valid_texts = [
                seg.text for seg in segments
                if seg.no_speech_prob < 0.60
            ]
            text = " ".join(valid_texts).strip()
            if text:
                self.text_queue.put(text)
        except Exception as e:
            print(f"[STT] Transcription failed: {e}")

    def stop(self):
        self.running = False
        print("[STT] Streaming STT worker stopped.")