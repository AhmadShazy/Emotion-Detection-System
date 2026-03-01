import os
import sys
import time
import queue
import threading
import subprocess
import pandas as pd
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Import the existing classification rules from V1
sys.path.append(PROJECT_ROOT)
from src.faceexpression.classifier import classify_emotion, smooth_emotions

class StreamingFace(threading.Thread):
    def __init__(self, face_queue, csv_path, openface_exe, poll_interval=1.0):
        """
        Worker thread for Streaming Face Expression Recognition.
        It launches OpenFace as a continuous subprocess and tails the resulting CSV file.
        openface_exe: path to FeatureExtraction.exe
        csv_path: expected output path for the CSV
        poll_interval: how often to check the CSV for new rows
        """
        super().__init__(daemon=True)
        self.face_queue = face_queue
        self.csv_path = csv_path
        self.openface_exe = openface_exe
        self.poll_interval = poll_interval
        
        self.running = False
        self.openface_process = None
        self.last_row_read = 0
        
        # We accumulate all emotions detected during the active Turn-Based speech
        self.recent_emotions = []
        
        # State to hold the final evaluated emotion for this turn
        self.current_emotion = {"emotion": None, "confidence": 0.0}

    def start_openface(self):
        print("[Face] Starting OpenFace Subprocess for continuous streaming...")
        out_dir = os.path.dirname(self.csv_path)
        out_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        cmd = [
            self.openface_exe,
            "-device", "0",
            "-out_dir", out_dir,
            "-of", out_name
        ]
        
        cwd = os.path.dirname(self.openface_exe)
        
        try:
            self.openface_process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Face] OpenFace running with PID: {self.openface_process.pid}")
        except Exception as e:
            print(f"[!] Failed to start OpenFace: {e}")
            self.running = False

    def run(self):
        self.running = True
        self.start_openface()
        
        print("[Face] Streaming Face worker started polling.")
        
        # Wait for CSV to be created (OpenFace on Windows can take up to 20-30s to boot)
        retries = 60
        while not os.path.exists(self.csv_path) and retries > 0 and self.running:
            time.sleep(1)
            retries -= 1
            
        if not os.path.exists(self.csv_path):
            print("[!] Face streaming failed: CSV file was never fully created.")
            self.running = False
            return
            
        while self.running:
            try:
                self._poll_csv()
            except Exception as e:
                print(f"Error reading Face CSV: {e}")
            time.sleep(self.poll_interval)
            
    def _poll_csv(self):
        """Reads only the new rows appended to the CSV file since the last poll."""
        try:
            # We skip the rows we've already read.
            # We use header=0 and skiprows to essentially skip previously processed rows efficiently
            # Note: pandas skiprows skips actual file lines, so we must account for header if skiprows > 0
            if self.last_row_read == 0:
                df = pd.read_csv(self.csv_path)
            else:
                # Read header then skip rows 1 to last_row_read
                skip = range(1, self.last_row_read + 1)
                df = pd.read_csv(self.csv_path, skiprows=skip)
                
            if df.empty:
                return
                
            # Clean columns as OpenFace has spaces
            df.columns = df.columns.str.strip()
            
            # Update index
            self.last_row_read += len(df)
            
            # Filter valid frames
            valid_df = df[(df["success"] == 1) & (df["confidence"] > 0.8)].copy()
            
            if valid_df.empty:
                return
                
            # Classify
            valid_df["emotion"] = valid_df.apply(classify_emotion, axis=1)
            
            # Add all detected frames to the active sentence pool
            for emo in valid_df["emotion"].tolist():
                self.recent_emotions.append(emo)
            
            if self.recent_emotions:
                # Count frequencies across the entire turn
                counts = Counter(self.recent_emotions)
                
                # Default to the most common emotion
                dominant_emotion = counts.most_common(1)[0][0]
                confidence = counts[dominant_emotion] / len(self.recent_emotions)
                
                # ==== ANTI-NEUTRAL & ANTI-SPEAKING FILTER ====
                # 1. Neutral Fallback: If Neutral is dominant but another emotion spiked, promote the spike.
                # 2. Speaking Mask (Happy) Fallback: Speaking forcefully pulls the lip corners (AU12), which 
                #    OpenFace falsely detects as 'Happy'. If Happy is dominant but Anger/Sadness/Surprise
                #    were detected in the background, we assume the 'Happy' was just articulation and promote the real emotion.
                if dominant_emotion in ["Neutral", "Happy"] and len(counts) > 1:
                    for emo, count in counts.items():
                        if emo not in ["Neutral", "Happy"] and count >= 1:
                            dominant_emotion = emo
                            # Artificial confidence boost to prioritize the underlying expression
                            confidence = 0.60 + (count * 0.1) 
                            break
                            
                self.current_emotion = {
                    "source": "face",
                    "emotion": dominant_emotion,
                    "confidence": confidence
                }
                
        except pd.errors.EmptyDataError:
            # File exists but is empty
            pass

    def stop(self):
        self.running = False
        if self.openface_process and self.openface_process.poll() is None:
            self.openface_process.terminate()
            try:
                self.openface_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.openface_process.kill()
        print("Stopped Streaming Face worker.")

    def get_current_emotion(self):
        return getattr(self, "current_emotion", {"emotion": None, "confidence": 0.0})

    def clear_buffer(self):
        """Flushes the accumulated face history so the next turn starts fresh."""
        self.recent_emotions = []
        self.current_emotion = {"emotion": None, "confidence": 0.0}
