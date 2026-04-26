import os
import sys
import pandas as pd
from collections import Counter

# ===============================
# 1. LOAD CSV
# ===============================
# Default path when run directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "live_session.csv")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        # Fallback for testing if running from different cwd
        CSV_PATH = r"C:\Users\ahmad\Desktop\humanoid-assistant-demo\data\processed\live_session.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    # If imported, we might not need to load immediately, but let's handle graceful init
    df = pd.DataFrame() 

if not df.empty:
    df.columns = df.columns.str.strip()  # 🔥 FIX OPENFACE COLUMN SPACES


# ===============================
# 2. FILTER VALID FRAMES
# ===============================
if not df.empty:
    df = df[(df["success"] == 1) & (df["confidence"] > 0.8)]
    df = df.reset_index(drop=True)

# if df.empty and __name__ == "__main__":
#    raise ValueError("No valid frames found after filtering!")


# ===============================
# 3. EMOTION CLASSIFICATION RULES
# ===============================
def classify_emotion(row):

    # Happy (Lip corner puller OR cheek raiser)
    if row["AU12_c"] == 1 or row["AU06_c"] == 1:
        return "Happy"

    # Angry (Brow lowerer AND Lid tightener) - Avoids false positives during speech
    if row["AU04_c"] == 1 and row["AU07_c"] == 1:
        return "Angry"

    # Sad (Inner brow raiser OR lip corner depressor)
    if row["AU01_c"] == 1 or row["AU15_c"] == 1:
        return "Sad"

    # Surprised (Inner brow raiser OR outer brow raiser OR jaw drop)
    if row["AU01_c"] == 1 or row["AU02_c"] == 1 or row["AU26_c"] == 1:
        return "Surprised"

    return "Neutral"

# ===============================
# 4. TEMPORAL SMOOTHING
# ===============================
def smooth_emotions(emotions, window=10):
    smoothed = []

    for i in range(len(emotions)):
        start = max(0, i - window)
        window_vals = emotions[start:i+1]
        dominant = Counter(window_vals).most_common(1)[0][0]
        smoothed.append(dominant)

    return smoothed

def analyze_openface_csv(csv_path):
    """
    Reads an OpenFace CSV, filters frames, classifies emotions, performs smoothing,
    saves the results, and prints the timeline.
    """
    print(f"📂 Analyzing: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return "File not found.", None

    df.columns = df.columns.str.strip()  # 🔥 FIX OPENFACE COLUMN SPACES

    # ===============================
    # 2. FILTER VALID FRAMES
    # ===============================
    df = df[(df["success"] == 1) & (df["confidence"] > 0.8)]
    df = df.reset_index(drop=True)

    if df.empty:
        print("⚠️ No valid frames found after filtering! (Confidence < 0.8)")
        return "No valid face frames detected.", None

    df["emotion"] = df.apply(classify_emotion, axis=1)
    df["smooth_emotion"] = smooth_emotions(df["emotion"], window=10)

    # ===============================
    # 5. SAVE FRAME-LEVEL OUTPUT
    # ===============================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    analysis_dir = os.path.join(project_root, "data", "analysis")
    
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    output_csv = os.path.join(analysis_dir, "frame_level_emotions.csv")
    df[["timestamp", "emotion", "smooth_emotion"]].to_csv(output_csv, index=False)
    print(f"📄 Saved frame-level data: {output_csv}")

    # ===============================
    # 6. GENERATE CLEAN EMOTION TIMELINE
    # ===============================
    segments = []

    current_emotion = df.loc[0, "smooth_emotion"]
    start_time = df.loc[0, "timestamp"]

    for i in range(1, len(df)):
        if df.loc[i, "smooth_emotion"] != current_emotion:
            end_time = df.loc[i - 1, "timestamp"]
            segments.append((start_time, end_time, current_emotion))

            current_emotion = df.loc[i, "smooth_emotion"]
            start_time = df.loc[i, "timestamp"]

    # last segment
    segments.append((start_time, df.loc[len(df) - 1, "timestamp"], current_emotion))

    # ===============================
    # 7. GENERATE FINAL CLEAN OUTPUT
    # ===============================
    final_output_path = os.path.join(analysis_dir, "final_emotions.txt")
    
    timeline_str = ""
    with open(final_output_path, "w") as f:
        for start, end, emo in segments:
            line = f"{start:.2f}s – {end:.2f}s : {emo}\n"
            f.write(line)
            timeline_str += line

    print(f"✅ Emotion extraction completed. Saved to {final_output_path}")
    
    # Calculate dominant emotion and confidence for the unified pipeline
    face_state = None
    if not df.empty:
        overall_counts = df["smooth_emotion"].value_counts(normalize=True)
        if not overall_counts.empty:
            dominant_emotion = overall_counts.index[0]
            confidence = float(overall_counts.iloc[0])
            
            # Count frequencies to calculate instability (matching Option 4)
            transitions = 0
            emotions_list = df["emotion"].tolist()
            for i in range(1, len(emotions_list)):
                if emotions_list[i] != emotions_list[i-1]:
                    transitions += 1
            instability = transitions / len(emotions_list) if len(emotions_list) > 1 else 0.0
            
            face_state = {
                "source": "face",
                "emotion": dominant_emotion,
                "confidence": confidence,
                "reliability": 1.0,
                "instability": instability
            }

    return timeline_str, face_state


if __name__ == "__main__":
    # Default behavior when run directly
    CSV_PATH = r"C:\Users\ahmad\Desktop\humanoid-assistant-demo\openface\OpenFace_2.2.0_win_x64\processed\live_session.csv"
    analyze_openface_csv(CSV_PATH)

