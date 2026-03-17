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
        self.current_emotion = {
            "source": "voice", "emotion": None, "confidence": 0.0, 
            "reliability": 0.0, "peak_emotion": None, "average_emotion": None
        }
        
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
        """Runs Wav2Vec2 on sliding window mini-segments for Peak + Average hybrid, then calculates reliability."""
        try:
            # 1. Energy Calculation (Volume)
            rms = np.sqrt(np.mean(self.audio_buffer**2))
            # Assume rms around 0.02 is reasonably audible speech, 0.005 is very quiet
            energy_score = min(1.0, rms / 0.02)
            
            label_map = {
                'hap': 'Happy',
                'ang': 'Angry',
                'neu': 'Neutral',
                'sad': 'Sad'
            }
            idx_to_label = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
            
            # Mini-chunk windowing (1 second chunks with 0.5s overlap)
            chunk_size = self.sample_rate  # 1 second
            step_size = self.sample_rate // 2 # 0.5 second overlap
            
            all_probs = []
            
            if len(self.audio_buffer) >= chunk_size:
                for start in range(0, len(self.audio_buffer) - chunk_size + 1, step_size):
                    end = start + chunk_size
                    chunk = self.audio_buffer[start:end]
                    tensor = torch.from_numpy(chunk).unsqueeze(0)
                    out_prob, _, _, _ = self.ser_engine.classifier.classify_batch(tensor)
                    probs = torch.softmax(out_prob[0], dim=0)
                    all_probs.append(probs)
            else:
                # If too small for sliding window, just eval once
                tensor = torch.from_numpy(self.audio_buffer).unsqueeze(0)
                out_prob, _, _, _ = self.ser_engine.classifier.classify_batch(tensor)
                probs = torch.softmax(out_prob[0], dim=0)
                all_probs.append(probs)

            # Stack into a 2D tensor: Shape (num_chunks, 4)
            stacked_probs = torch.stack(all_probs)
            
            # 1. Average Emotion Logic
            avg_probs = torch.mean(stacked_probs, dim=0)
            avg_top_probs, avg_top_indices = torch.topk(avg_probs, 2)
            
            avg_emotion = idx_to_label.get(int(avg_top_indices[0]), 'Neutral')
            avg_conf = float(avg_top_probs[0])
            
            # Anti-neutral logic for Average
            if avg_emotion == 'Neutral' and avg_conf < 0.60:
                sec_emotion = idx_to_label.get(int(avg_top_indices[1]), 'Neutral')
                if sec_emotion != 'Neutral':
                    avg_emotion = sec_emotion
                    avg_conf = float(avg_top_probs[1])
                    
            # 2. Peak Emotion Logic (Find the highest non-neutral spike across chunks)
            peak_emotion = avg_emotion
            peak_conf = 0.0
            
            for probs in all_probs:
                top_p, top_idx = torch.topk(probs, 2)
                em = idx_to_label.get(int(top_idx[0]), 'Neutral')
                conf = float(top_p[0])
                
                # Favor non-neutral spikes
                if em != 'Neutral' and conf > peak_conf:
                    peak_emotion = em
                    peak_conf = conf
                elif em == 'Neutral' and conf < 0.60:
                    sec_em = idx_to_label.get(int(top_idx[1]), 'Neutral')
                    if sec_em != 'Neutral' and float(top_p[1]) > peak_conf:
                        peak_emotion = sec_em
                        peak_conf = float(top_p[1])
                        
            # Fallback if no non-neutral spikes found
            if peak_conf == 0.0:
                peak_emotion = avg_emotion
                peak_conf = avg_conf

            # 3. Hybrid Fusion
            final_emotion = peak_emotion
            final_conf = (0.6 * avg_conf) + (0.4 * peak_conf)
            if peak_emotion != avg_emotion:
                # If there's a strong non-neutral peak that differs from a neutral average, promote the peak slightly
                if avg_emotion == 'Neutral' and peak_conf > 0.5:
                    final_emotion = peak_emotion
                    final_conf = peak_conf * 0.9 # Small decay to reflect it was only a peak
                else:
                    # Let average win if multiple non-neutral emotions conflict
                    final_emotion = avg_emotion
                    final_conf = avg_conf

            # 4. Reliability Calculation
            # Spread: Difference between top 2 probs in the average
            prob_spread = float(avg_top_probs[0] - avg_top_probs[1])
            spread_score = min(1.0, prob_spread * 2.0) # Map [0.0, 0.5] prob spread to [0.0, 1.0]
            
            # Combine energy and spread into reliability
            # Only highly energized speech with a clear distribution is 100% reliable
            reliability = (0.5 * energy_score) + (0.5 * spread_score)

            self.current_emotion = {
                "source": "voice",
                "emotion": final_emotion,
                "confidence": final_conf,
                "average_emotion": avg_emotion,
                "peak_emotion": peak_emotion,
                "reliability": reliability
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
            
        return getattr(self, "current_emotion", {
            "source": "voice", "emotion": None, "confidence": 0.0, 
            "reliability": 0.0, "peak_emotion": None, "average_emotion": None
        })

    def clear_buffer(self):
        """Flushes the accumulated audio history so the next turn starts fresh."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_emotion = {
            "source": "voice", "emotion": None, "confidence": 0.0, 
            "reliability": 0.0, "peak_emotion": None, "average_emotion": None
        }
