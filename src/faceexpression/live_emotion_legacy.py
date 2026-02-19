import subprocess
import pandas as pd
import time
import os
from collections import Counter

# ===============================
# CONFIG
# ===============================
OPENFACE_EXE = r"C:\Users\ahmad\Desktop\humanoid-assistant-demo\openface\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

OUTPUT_DIR = "processed"
CSV_NAME = "live_session.csv"

CSV_PATH = os.path.join(OUTPUT_DIR, CSV_NAME)

# ===============================
# EMOTION LOGIC
# ===============================
def classify_emotion(row):
    if row["AU12_c"] == 1 and row["AU06_c"] == 1:
        return "Happy"
    if row["AU04_c"] == 1 and row["AU07_c"] == 1 and row["AU23_c"] == 1:
        return "Angry"
    if row["AU01_c"] == 1 and row["AU04_c"] == 1 and row["AU15_c"] == 1:
        return "Sad"
    if row["AU01_c"] == 1 and row["AU02_c"] == 1 and row["AU26_c"] == 1:
        return "Surprised"
    return "Neutral"

# ===============================
# START OPENFACE
# ===============================
print("▶ Starting OpenFace...")

openface_cmd = [
    OPENFACE_EXE,
    "-device", "0",
    "-out_dir", OUTPUT_DIR,
    "-of", "live_session"
]

process = subprocess.Popen(openface_cmd)

# ===============================
# WAIT FOR CSV TO APPEAR
# ===============================
print("⏳ Waiting for CSV...")
while not os.path.exists(CSV_PATH):
    time.sleep(0.2)

print("✅ CSV detected. Live emotion tracking started.\n")

# ===============================
# LIVE CSV MONITORING
# ===============================
last_row = 0
emotion_buffer = []

try:
    while process.poll() is None:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()

        if len(df) > last_row:
            new_rows = df.iloc[last_row:]
            last_row = len(df)

            for _, row in new_rows.iterrows():
                if row["success"] != 1 or row["confidence"] < 0.8:
                    continue

                emo = classify_emotion(row)
                emotion_buffer.append(emo)

                # smooth over last 10 frames
                if len(emotion_buffer) >= 10:
                    dominant = Counter(emotion_buffer[-10:]).most_common(1)[0][0]
                    print(f"🧠 Emotion: {dominant}")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

finally:
    process.terminate()
    print("✔ OpenFace terminated.")
