import threading
import queue
import time
import numpy as np
from faster_whisper import WhisperModel

class StreamingSTT(threading.Thread):
    def __init__(self, audio_queue, text_queue, status_queue=None, model_size="tiny", compute_type="int8", 
                 silence_threshold=0.01, trailing_silence_seconds=2.0, sample_rate=16000):
        """
        Worker thread for Streaming Speech-To-Text using faster-whisper on CPU.
        - audio_queue: queue to read audio chunks from
        - text_queue: queue to push transcribed text to
        - status_queue: OPTIONAL queue to push VAD state strings ("LISTENING", "ANALYZING")
        - silence_threshold: RMS amplitude below which is considered silence
        - trailing_silence_seconds: Seconds of continuous silence required to trigger the end of a sentence.
        """
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.status_queue = status_queue
        
        self.silence_threshold = silence_threshold
        self.trailing_silence_frames = int(trailing_silence_seconds * sample_rate)
        self.current_silence_frames = 0
        self.sample_rate = sample_rate
        
        self.running = False
        self.audio_buffer = np.array([], dtype=np.float32)

        print(f"Loading faster-whisper '{model_size}' model ...")
        # Initialize model (cpu int8 is very fast)
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        print("faster-whisper loaded.")

    def run(self):
        self.running = True
        print("[STT] Streaming STT worker started.")
        has_spoken = False
        
        while self.running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.5)
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                
                # Check for silence
                is_silent = self._is_silent(chunk)
                
                if is_silent:
                    self.current_silence_frames += len(chunk)
                else:
                    self.current_silence_frames = 0
                    if not has_spoken:
                        has_spoken = True
                        if self.status_queue:
                            self.status_queue.put("LISTENING")
                
                # If we detect enough trailing silence
                if self.current_silence_frames >= self.trailing_silence_frames:
                    if has_spoken:
                        if self.status_queue:
                            self.status_queue.put("ANALYZING")
                        # Transcription triggered by natural spoken pause
                        self._transcribe_buffer()
                        has_spoken = False
                    else:
                        # It's just background silence; flush buffer to prevent memory bloat
                        self.audio_buffer = np.array([], dtype=np.float32)
                        
                    self.current_silence_frames = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"STT Worker Error: {e}")

    def _is_silent(self, chunk):
        """Checks if the RMS amplitude of the chunk is below the silence threshold."""
        rms = np.sqrt(np.mean(chunk**2))
        return rms < self.silence_threshold

    def _transcribe_buffer(self):
        """Runs faster-whisper on the accumulated buffer."""
        # 1.0 second minimum of audio required to attempt transcription (prevents mic bumps)
        if len(self.audio_buffer) < self.sample_rate: 
            self.audio_buffer = np.array([], dtype=np.float32)
            return

        # Prepare audio for transcription
        audio_data = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.float32) # Clear buffer

        try:
            segments, info = self.model.transcribe(audio_data, beam_size=5)
            
            # Filter hallucinations: only keep segments where the model is confident someone is actually speaking
            valid_texts = []
            for segment in segments:
                if segment.no_speech_prob < 0.60:
                    valid_texts.append(segment.text)
            
            text = " ".join(valid_texts).strip()
            
            if text:
                self.text_queue.put(text)
        except Exception as e:
            print(f"Transcription failed: {e}")

    def stop(self):
        self.running = False
        print("Stopped Streaming STT worker.")

if __name__ == "__main__":
    # Simple test setup
    import sounddevice as sd
    audio_q = queue.Queue()
    text_q = queue.Queue()
    
    stt = StreamingSTT(audio_q, text_q)
    stt.start()
    
    # Record some audio manually and push to queue
    print("Recording 3 seconds...")
    audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    audio_q.put(audio[:, 0])
    
    print("Waiting for transcription...")
    try:
        text = text_q.get(timeout=5.0)
        print(f"Final Test output: {text}")
    except queue.Empty:
        print("No transcription generated.")
        
    stt.stop()
    stt.join()
