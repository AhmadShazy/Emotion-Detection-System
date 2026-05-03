import os
import sys
import time
import queue
import threading
import subprocess
import pandas as pd
from collections import Counter

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(PROJECT_ROOT)
from src.faceexpression.classifier import classify_emotion, smooth_emotions


class StreamingFace(threading.Thread):
    def __init__(self, face_queue, csv_path, openface_exe, poll_interval=1.0):
        """
        Worker thread for streaming face expression recognition.

        Launches OpenFace as a continuous subprocess and incrementally reads
        the resulting CSV file to track emotions in real time.

        Parameters
        ----------
        face_queue    : queue.Queue or None — legacy; not used internally anymore.
        csv_path      : str  — expected output path for OpenFace CSV.
        openface_exe  : str  — full path to FeatureExtraction.exe.
        poll_interval : float — seconds between CSV polls.
        """
        super().__init__(daemon=True)
        self.face_queue     = face_queue
        self.csv_path       = csv_path
        self.openface_exe   = openface_exe
        self.poll_interval  = poll_interval

        self.running          = False
        self.openface_process = None

        # ── Incremental read state ────────────────────────────────────────────
        # Tracks how many DATA rows (excluding header) we have already processed.
        # We read the full CSV each poll and slice with iloc[self._rows_read:]
        # so column names are always parsed correctly from row 0.
        self._rows_read = 0

        # ── Per-turn emotion accumulator ──────────────────────────────────────
        self.recent_emotions      = []
        self.total_frames_polled  = 0
        self.valid_frames_polled  = 0

        self.current_emotion = {
            "emotion": None, "confidence": 0.0,
            "reliability": 0.0, "instability": 0.0,
        }

    # ── OpenFace subprocess ───────────────────────────────────────────────────

    def start_openface(self):
        print("[Face] Starting OpenFace subprocess for continuous streaming...")
        out_dir  = os.path.dirname(self.csv_path)
        out_name = os.path.splitext(os.path.basename(self.csv_path))[0]

        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            self.openface_exe,
            "-device", "0",
            "-out_dir", out_dir,
            "-of", out_name,
        ]
        cwd = os.path.dirname(self.openface_exe)

        try:
            self.openface_process = subprocess.Popen(
                cmd, cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[Face] OpenFace running (PID {self.openface_process.pid})")
        except Exception as e:
            print(f"[Face] ❌ Failed to start OpenFace: {e}")
            self.running = False

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self.start_openface()

        # Wait up to 60 s for OpenFace to create the CSV file
        retries = 60
        while not os.path.exists(self.csv_path) and retries > 0 and self.running:
            time.sleep(1)
            retries -= 1

        if not os.path.exists(self.csv_path):
            print("[Face] ❌ CSV never appeared — face streaming disabled.")
            self.running = False
            return

        print("[Face] CSV detected. Streaming face worker now polling.")

        while self.running:
            try:
                self._poll_csv()
            except Exception as e:
                print(f"[Face] Poll error: {e}")
            time.sleep(self.poll_interval)

    # ── Incremental CSV reader ────────────────────────────────────────────────

    def _poll_csv(self):
        """
        Reads only new rows appended since the last poll.

        Fix for Bug 5:
        Previously used skiprows=range(1, N) which skips row *numbers*
        in the file — so the header row (index 0) was read as data on
        subsequent polls, corrupting column names.

        Correct approach: always read the full file with a consistent header,
        then slice off already-processed rows using iloc[self._rows_read:].
        This is slightly less efficient for huge CSVs but OpenFace CSVs are
        small (one row per video frame at 30fps) so it's perfectly fine.
        """
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            return  # File exists but OpenFace hasn't written anything yet
        except Exception:
            return

        # Strip whitespace from column names (OpenFace adds leading spaces)
        df.columns = df.columns.str.strip()

        # Slice to only the NEW rows since last poll
        new_rows = df.iloc[self._rows_read:]
        if new_rows.empty:
            return

        self._rows_read      += len(new_rows)
        self.total_frames_polled += len(new_rows)

        # Filter: only confident, successfully tracked frames
        valid = new_rows[
            (new_rows["success"] == 1) & (new_rows["confidence"] > 0.8)
        ].copy()

        self.valid_frames_polled += len(valid)

        if valid.empty:
            return

        # Classify each valid frame
        valid["emotion"] = valid.apply(classify_emotion, axis=1)

        for emo in valid["emotion"].tolist():
            self.recent_emotions.append(emo)

        self._update_current_emotion()

    # ── Emotion state updater ─────────────────────────────────────────────────

    def _update_current_emotion(self):
        if not self.recent_emotions:
            return

        counts           = Counter(self.recent_emotions)
        dominant_emotion = counts.most_common(1)[0][0]
        confidence       = counts[dominant_emotion] / len(self.recent_emotions)

        # ── Anti-neutral / anti-speaking-mask filter ──────────────────────────
        # Speaking forcefully pulls lip corners (AU12), which OpenFace falsely
        # reads as "Happy". If Happy or Neutral is dominant but another emotion
        # spiked, promote the spike instead.
        if dominant_emotion in ("Neutral", "Happy") and len(counts) > 1:
            for emo, count in counts.items():
                if emo not in ("Neutral", "Happy") and count >= 1:
                    dominant_emotion = emo
                    confidence       = min(0.95, 0.60 + count * 0.02)
                    break

        # ── Reliability ───────────────────────────────────────────────────────
        reliability = (
            self.valid_frames_polled / self.total_frames_polled
            if self.total_frames_polled > 0 else 0.0
        )

        # ── Instability ───────────────────────────────────────────────────────
        transitions = sum(
            1 for i in range(1, len(self.recent_emotions))
            if self.recent_emotions[i] != self.recent_emotions[i - 1]
        )
        instability = (
            transitions / len(self.recent_emotions)
            if len(self.recent_emotions) > 1 else 0.0
        )

        self.current_emotion = {
            "source":      "face",
            "emotion":     dominant_emotion,
            "confidence":  confidence,
            "reliability": reliability,
            "instability": instability,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_emotion(self):
        return self.current_emotion

    def clear_buffer(self):
        """Flush accumulated history so the next turn starts fresh."""
        self.recent_emotions      = []
        self.total_frames_polled  = 0
        self.valid_frames_polled  = 0
        self.current_emotion = {
            "emotion": None, "confidence": 0.0,
            "reliability": 0.0, "instability": 0.0,
        }

    def stop(self):
        self.running = False
        if self.openface_process and self.openface_process.poll() is None:
            self.openface_process.terminate()
            try:
                self.openface_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.openface_process.kill()
        print("[Face] Streaming face worker stopped.")