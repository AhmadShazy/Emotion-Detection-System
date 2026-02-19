import subprocess
import os
import sys
import time

# ===============================
# CONFIG
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

OPENFACE_DIR = os.path.join(PROJECT_ROOT, "external", "openface", "OpenFace_2.2.0_win_x64")
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_FILENAME = "live_session"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.csv")

# Ensure the classifier script is in the python path
sys.path.append(SCRIPT_DIR)

try:
    from classifier import analyze_openface_csv
except ImportError as e:
    print(f"❌ Error: Could not import 'classifier.py'. Details: {e}")
    sys.exit(1)

def run_face_pipeline():
    print("==========================================")
    print("🎥 STARTING OPENFACE EMOTION PIPELINE")
    print("==========================================")

    # 1. Start OpenFace
    print(f"▶ Launching OpenFace...")
    print(f"   - Device: Webcam 0")
    print(f"   - Output: {OUTPUT_CSV_PATH}")
    print("\n⚠️  IMPORTANT: A window will open showing the camera feed.")
    print("🛑  CLOSE THE WINDOW or PRESS 'Q' IN THE WINDOW TO STOP RECORDING.\n")
    
    cmd = [
        OPENFACE_EXE,
        "-device", "0",
        "-out_dir", OUTPUT_DIR,
        "-of", OUTPUT_FILENAME
    ]

    try:
        # Run OpenFace and wait for it to finish
        process = subprocess.run(cmd, cwd=OPENFACE_DIR)
        
        if process.returncode != 0:
            print(f"\n⚠️ OpenFace exited with code {process.returncode}. Proceeding to analysis anyway...")
        else:
            print("\n✅ OpenFace recording finished.")

    except FileNotFoundError:
        print(f"\n❌ Error: Could not find OpenFace executable at: {OPENFACE_EXE}")
        return
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline stopped by user.")
        return

    # 2. Run Analysis
    print("\n==========================================")
    print("🧠 ANALYZING RECORDED DATA")
    print("==========================================")
    
    if os.path.exists(OUTPUT_CSV_PATH):
        # Give a small buffer for file IO to close completely
        time.sleep(1)
        analyze_openface_csv(OUTPUT_CSV_PATH)
    else:
        print(f"❌ Error: Expected output CSV not found: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_face_pipeline()

