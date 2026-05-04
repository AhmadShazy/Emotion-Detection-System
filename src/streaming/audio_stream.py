import time
import queue
import numpy as np
import sounddevice as sd


class AudioStreamer:
    def __init__(self, sample_rate=16000, chunk_duration=0.25):
        """
        Non-blocking audio stream broadcaster.

        sample_rate:    16000 Hz — matches Whisper and Wav2Vec2.
        chunk_duration: 0.25s per chunk — 250ms buffer per callback.
        """
        self.sample_rate     = sample_rate
        self.chunk_duration  = chunk_duration
        self.blocksize       = int(self.sample_rate * self.chunk_duration)

        self.audio_queues    = []
        self.stream          = None
        self._overflow_count = 0

    def add_queue(self, q):
        """Registers a queue to receive a copy of the audio stream."""
        self.audio_queues.append(q)

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice for each audio block.
        Runs in a C-thread — must be fast and non-blocking.
        """
        if status:
            self._overflow_count += 1
            # Log every 10th overflow to avoid console spam
            if self._overflow_count % 10 == 1:
                print(
                    f"[Audio] ⚠️  Stream overflow x{self._overflow_count} "
                    f"— STT/SER processing may be too slow for real-time."
                )

        # Flatten (frames, 1) → (frames,) float32
        audio_chunk = indata[:, 0].copy()

        for q in self.audio_queues:
            try:
                q.put_nowait(audio_chunk)
            except queue.Full:
                # Queue full — drop oldest chunk to make room for fresh audio
                # Fresh audio is always preferred over stale audio
                try:
                    q.get_nowait()
                    q.put_nowait(audio_chunk)
                except Exception:
                    pass

    def start(self):
        """Starts the non-blocking audio input stream."""
        if self.stream is not None:
            return

        print("[Audio] Starting continuous audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self):
        """Stops the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("[Audio] Audio stream stopped.")

    def clear_queues(self):
        self.audio_queues = []


if __name__ == "__main__":
    streamer = AudioStreamer(chunk_duration=1.0)
    q1       = queue.Queue()
    streamer.add_queue(q1)
    streamer.start()

    print("Listening for 5 seconds...")
    try:
        for _ in range(5):
            chunk = q1.get(timeout=2.0)
            print(
                f"Received chunk: shape={chunk.shape}, "
                f"max_amp={np.max(np.abs(chunk)):.4f}"
            )
    except queue.Empty:
        print("Timeout waiting for audio.")
    finally:
        streamer.stop()