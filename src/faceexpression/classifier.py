import os
import sys
import pandas as pd
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")

# NOTE: No top-level pd.read_csv() here.
# Previously this file ran pd.read_csv() at import time which crashed
# FastAPI startup because the CSV doesn't exist until a recording is made.
# All CSV reading is now done inside analyze_openface_csv() only.


# ── Emotion Classification Rules ──────────────────────────────────────────────

def classify_emotion(row):
    """
    Classifies a single OpenFace frame row into an emotion
    based on Action Unit (AU) presence flags.

    Rule ordering matters — each check must be mutually exclusive:
        Happy     → AU12_c (lip corner puller) OR AU06_c (cheek raiser)
        Surprised → AU26_c (jaw drop) OR (AU01_c AND AU02_c — both brows raised)
        Angry     → AU04_c AND AU07_c (brow lower + lid tighten, both required)
        Sad       → AU01_c AND AU15_c (inner brow raise + lip corner depressor)
                    OR AU15_c alone (lip depressor is a reliable sad marker)
        Neutral   → fallback

    Bug MM2 fix:
        Previously AU01_c (inner brow raise) triggered BOTH the Sad and
        Surprised rules. Since Sad came first, Surprised was unreachable via
        AU01_c. Fixed by:
          1. Requiring AU26_c (jaw drop) OR the AU01+AU02 combination for Surprised.
          2. Requiring BOTH AU01_c AND AU15_c for Sad (not AU01 alone).
          3. AU15_c alone as a reliable fallback for Sad.
    """
    # Happy — lip corner puller OR cheek raiser
    if row["AU12_c"] == 1 or row["AU06_c"] == 1:
        return "Happy"

    # Surprised — jaw drop (most reliable) OR both brows simultaneously raised
    # AU26_c alone is sufficient; AU01+AU02 together indicate full brow raise
    if row["AU26_c"] == 1 or (row["AU01_c"] == 1 and row["AU02_c"] == 1):
        return "Surprised"

    # Angry — brow lowerer AND lid tightener (both required to avoid false positives)
    if row["AU04_c"] == 1 and row["AU07_c"] == 1:
        return "Angry"

    # Sad — inner brow raise AND lip corner depressor together
    if row["AU01_c"] == 1 and row["AU15_c"] == 1:
        return "Sad"

    # Sad fallback — lip corner depressor alone is a reliable sad indicator
    if row["AU15_c"] == 1:
        return "Sad"

    return "Neutral"


# ── Temporal Smoothing ────────────────────────────────────────────────────────

def smooth_emotions(emotions, window=10):
    """
    Applies a sliding-window majority vote to reduce per-frame noise.
    Each frame is relabelled with the dominant emotion in the last `window` frames.
    """
    smoothed = []
    for i in range(len(emotions)):
        start       = max(0, i - window)
        window_vals = emotions[start : i + 1]
        dominant    = Counter(window_vals).most_common(1)[0][0]
        smoothed.append(dominant)
    return smoothed


# ── Main Analysis Function ────────────────────────────────────────────────────


# ── Note ─────────────────────────────────────────────────────────────────────
# analyze_openface_csv() lived here and read an OpenFace CSV produced by the
# server's own webcam. That mode is gone: video now arrives from the browser and
# is read by src/faceexpression/mediapipe_analyzer.py instead.
#
# It also wrote two FIXED paths (data/analysis/frame_level_emotions.csv and
# final_emotions.txt), so two concurrent requests overwrote each other and could
# raise PermissionError on Windows from a request whose analysis had succeeded.
# The replacement writes nothing to disk at all.
#
# classify_emotion and smooth_emotions above are still used by
# src/streaming/streaming_face.py (live stream), which moves to MediaPipe in
# Phase 4.
