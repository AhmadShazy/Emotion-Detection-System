import time
import queue
import numpy as np
import sounddevice as sd

class AudioStreamer:
    def __init__(self, sample_rate=16000, chunk_duration=1.0):
        """
        Initializes the non-blocking audio stream using sounddevice.
        - sample_rate: 16000 Hz (standard for whisper/wav2vec)
        - chunk_duration: Time in seconds per chunk (1.0 = 1 sec buffer)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.blocksize = int(self.sample_rate * self.chunk_duration)
        
        # We'll push numpy arrays into these registered queues
        self.audio_queues = []
        self.stream = None
        
    def add_queue(self, q):
        """Registers a queue to receive a copy of the audio stream."""
        self.audio_queues.append(q)

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback invoked by sounddevice for each audio block.
        Runs in a separate C-thread, must be fast and non-blocking.
        """
        if status:
            print(f"[!] Audio Stream Status: {status}")
            
        # indata is shape (frames, channels), e.g., (16000, 1)
        # We flatten it to a 1D array of float32 for downstream tasks
        audio_chunk = indata[:, 0].copy()
        
        # Push to all registered queues (broadcasting)
        for q in self.audio_queues:
            try:
                q.put_nowait(audio_chunk)
            except queue.Full:
                pass # Depending on architecture, silent drop can be intentional

    def start(self):
        """Starts the non-blocking audio stream."""
        if self.stream is not None:
            return

        print("[Audio] Starting continuous audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.blocksize,
            callback=self._audio_callback
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

# Simple test block
if __name__ == "__main__":
    streamer = AudioStreamer(chunk_duration=1.0)
    q1 = queue.Queue()
    streamer.add_queue(q1)
    streamer.start()
    
    print("Listening for 5 seconds...")
    try:
        for _ in range(5):
            chunk = q1.get(timeout=2.0)
            print(f"Received chunk: shape={chunk.shape}, max_amp={np.max(np.abs(chunk)):.4f}")
    except queue.Empty:
        print("Timeout waiting for audio.")
    finally:
        streamer.stop()
