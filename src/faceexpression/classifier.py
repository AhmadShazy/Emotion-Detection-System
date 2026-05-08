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

def analyze_openface_csv(csv_path):
    """
    Reads an OpenFace CSV, filters valid frames, classifies emotions,
    applies temporal smoothing, saves per-frame and timeline outputs,
    and returns (timeline_str, face_state).

    Returns:
        (str, dict | None)  — timeline string and face_state dict,
                               or ("error message", None) on failure.
    """
    print(f"📂 Analyzing: {csv_path}")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return "File not found.", None
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return f"CSV read error: {e}", None

    # Strip whitespace from column names (OpenFace adds leading spaces)
    df.columns = df.columns.str.strip()

    # ── Filter Valid Frames ───────────────────────────────────────────────────
    df = df[(df["success"] == 1) & (df["confidence"] > 0.8)].reset_index(drop=True)

    if df.empty:
        print("⚠️  No valid frames found after filtering (confidence < 0.8)")
        return "No valid face frames detected.", None

    # ── Classify + Smooth ─────────────────────────────────────────────────────
    df["emotion"]        = df.apply(classify_emotion, axis=1)
    df["smooth_emotion"] = smooth_emotions(df["emotion"].tolist(), window=10)

    # ── Save Frame-Level Output ───────────────────────────────────────────────
    analysis_dir = os.path.join(PROJECT_ROOT, "data", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    output_csv = os.path.join(analysis_dir, "frame_level_emotions.csv")
    df[["timestamp", "emotion", "smooth_emotion"]].to_csv(output_csv, index=False)
    print(f"📄 Saved frame-level data: {output_csv}")

    # ── Build Timeline ────────────────────────────────────────────────────────
    segments = []
    current_emotion = df.loc[0, "smooth_emotion"]
    start_time      = df.loc[0, "timestamp"]

    for i in range(1, len(df)):
        if df.loc[i, "smooth_emotion"] != current_emotion:
            segments.append((start_time, df.loc[i - 1, "timestamp"], current_emotion))
            current_emotion = df.loc[i, "smooth_emotion"]
            start_time      = df.loc[i, "timestamp"]

    segments.append((start_time, df.loc[len(df) - 1, "timestamp"], current_emotion))

    # ── Save Timeline ─────────────────────────────────────────────────────────
    final_output_path = os.path.join(analysis_dir, "final_emotions.txt")
    timeline_str = ""
    with open(final_output_path, "w") as f:
        for start, end, emo in segments:
            line = f"{start:.2f}s – {end:.2f}s : {emo}\n"
            f.write(line)
            timeline_str += line

    print(f"✅ Emotion extraction completed. Saved to {final_output_path}")

    # ── Build face_state for Unified Pipeline ─────────────────────────────────
    face_state = None
    overall_counts = df["smooth_emotion"].value_counts(normalize=True)

    if not overall_counts.empty:
        dominant_emotion = overall_counts.index[0]
        confidence       = float(overall_counts.iloc[0])

        # Instability = fraction of frames where emotion changed vs previous frame
        emotions_list = df["emotion"].tolist()
        transitions   = sum(
            1 for i in range(1, len(emotions_list))
            if emotions_list[i] != emotions_list[i - 1]
        )
        instability = transitions / len(emotions_list) if len(emotions_list) > 1 else 0.0

        face_state = {
            "source":      "face",
            "emotion":     dominant_emotion,
            "confidence":  confidence,
            "reliability": 1.0,
            "instability": instability,
        }

    return timeline_str, face_state


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH = r"C:\Users\ahmad\Desktop\humanoid-assistant-demo\openface\OpenFace_2.2.0_win_x64\processed\live_session.csv"
    timeline, state = analyze_openface_csv(CSV_PATH)
    print(timeline)
    print(state)