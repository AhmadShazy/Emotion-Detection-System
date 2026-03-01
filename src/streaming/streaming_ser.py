import threading
import queue
import time
import numpy as np
import torch
import sys
import os

# We reuse the SER Engine to load the model and patch torchaudio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ser.ser_engine import SEREngine

class StreamingSER(threading.Thread):
    def __init__(self, audio_queue, emotion_queue=None, sample_rate=16000):
        """
        Worker thread for Streaming Speech Emotion Recognition.
        Maintains a dynamically growing audio buffer, and evaluates the whole sequence when requested.
        """
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.emotion_queue = emotion_queue
        self.sample_rate = sample_rate
        
        self.running = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_emotion = {"emotion": None, "confidence": 0.0}
        
        # Load model once at startup!
        print("Loading SER Engine for streaming...")
        self.ser_engine = SEREngine()
        print("SER Engine loaded.")

    def run(self):
        self.running = True
        print("[SER] Streaming SER worker started.")
        
        while self.running:
            try:
                # Wait for audio chunks from the queue
                chunk = self.audio_queue.get(timeout=0.5)
                # Accumulate endlessly until the orchestrator clears the buffer (dynamically sized)
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"SER Worker Error: {e}")

    def _classify_buffer(self):
        """Runs Wav2Vec2 on the current sliding window buffer."""
        # Convert buffer to tensor shape expected by classifier: (1, time)
        tensor = torch.from_numpy(self.audio_buffer).unsqueeze(0)
        
        try:
            # We bypass _custom_load and directly pass the tensor
            out_prob, score, index, text_lab = self.ser_engine.classifier.classify_batch(tensor)
            
            label_map = {
                'hap': 'Happy',
                'ang': 'Angry',
                'neu': 'Neutral',
                'sad': 'Sad'
            }
            
            # The tensor out_prob[0] contains the log-probs or logits for the 4 classes
            probs = torch.softmax(out_prob[0], dim=0)
            
            # Get values and indices of the top 2 emotions
            top_probs, top_indices = torch.topk(probs, 2)
            
            # The model's classes are ['ang', 'hap', 'neu', 'sad'] in index 0, 1, 2, 3 natively by speechbrain
            # We must map the numeric indices back to our textual labels via the engine's label dict
            # Luckily, text_lab gives us the top-1 string label automatically.
            raw_label = text_lab[0]
            emotion = label_map.get(raw_label, raw_label)
            confidence = float(top_probs[0])
            
            # ==== NEUTRAL PENALTY FILTER ====
            # Wav2Vec2 is wildly biased toward 'Neutral' in noisy environments.
            # If the top prediction is Neutral, but it is weakly confident (< 0.60), 
            # we drop it and use the 2nd highest emotion instead to capture the true undertone.
            if emotion == 'Neutral' and confidence < 0.60:
                # We need to map top_indices[1] to a string. 
                # According to HF IEMOCAP label mapping: 0=ang, 1=hap, 2=neu, 3=sad
                idx_to_label = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
                second_idx = int(top_indices[1])
                second_emotion = idx_to_label.get(second_idx, 'Neutral')
                
                # If the second emotion isn't ALSO neutral, we promote it!
                if second_emotion != 'Neutral':
                    emotion = second_emotion
                    confidence = float(top_probs[1])
            
            # print(f"  [SER] {emotion} ({confidence:.2f})")
            
            self.current_emotion = {
                "source": "voice",
                "emotion": emotion,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"SER classification error: {e}")

    def stop(self):
        self.running = False
        print("Stopped Streaming SER worker.")

    def get_current_emotion(self):
        """Classify the entire dynamically accumulated sentence buffer on demand."""
        if len(self.audio_buffer) >= self.sample_rate: # Need at least 1 second to classify safely
            self._classify_buffer()
            
        return getattr(self, "current_emotion", {"emotion": None, "confidence": 0.0})

    def clear_buffer(self):
        """Flushes the accumulated audio history so the next turn starts fresh."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_emotion = {"emotion": None, "confidence": 0.0}
