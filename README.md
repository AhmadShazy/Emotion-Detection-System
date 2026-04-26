# Humanoid Assistant Demo

A modular, multimodal emotion-analysis system capable of analyzing emotion from **text**, **voice**, **facial expressions**, and all three simultaneously in real-time.

---

## Project Structure

```
humanoid-assistant-demo/
├── main.py                         # Entry point — main menu
├── requirements.txt
├── external/
│   ├── openface/
│   │   └── OpenFace_2.2.0_win_x64/
│   │       └── FeatureExtraction.exe   # Required external binary
│   └── whisper/                    # Whisper model cache (auto-downloaded)
├── data/
│   ├── recordings/                 # Saved .wav files
│   ├── processed/                  # OpenFace CSV outputs
│   └── analysis/                   # frame_level_emotions.csv, final_emotions.txt
└── src/
    ├── faceexpression/             # OpenFace pipeline + AU-based classifier
    ├── ser/                        # SpeechBrain Wav2Vec2 SER engine
    ├── stt/                        # Whisper speech-to-text
    ├── text_emotion/               # RoBERTa text emotion (go_emotions)
    ├── full_analysis/              # Legacy batch mode (all 3 pipelines)
    └── streaming/                  # V2 real-time streaming architecture
```

---

## Setup & Requirements

1. **Python Environment** — Activate the project virtual environment (`.venv`).
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries: `speechbrain`, `transformers`, `torch`, `openai-whisper`, `faster-whisper`, `sounddevice`, `numpy`, `pandas`, `soundfile`

3. **OpenFace Binary** — Place the executable at:
   ```
   external/openface/OpenFace_2.2.0_win_x64/FeatureExtraction.exe
   ```

4. **Model Downloads** — Whisper, SpeechBrain Wav2Vec2-IEMOCAP, and RoBERTa go_emotions are downloaded automatically on first use.

---

## Usage

Run the main menu:

```bash
python main.py
```

---

## Menu Options

```
==========================================
   HUMANOID ASSISTANT - MAIN MENU
==========================================
  1. 💬 Text Emotion Analysis  (Keyboard Input)
  2. 🎤 Voice Analysis          (Speech + Emotion)
  3. 🎬 Multimodal Recording    (Video + Voice Analysis)
  4. 🌐 Live Multimodal Chat    (Real-Time Streaming)
  5. 🚪 Exit
==========================================
```

### Option 1 — Text Emotion Analysis
- Type any text directly at the keyboard
- Analyzes emotion using **RoBERTa** (`SamLowe/roberta-base-go_emotions`)
- Displays top emotion labels with confidence scores and a visual bar chart
- Loops so you can analyze multiple texts without returning to the menu

### Option 2 — Voice Analysis
Records your voice **once** and runs three analyses on the same audio file:

| Step | Model | Output |
|---|---|---|
| 1 | **SpeechBrain Wav2Vec2** | Voice emotion (Happy / Angry / Neutral / Sad) |
| 2 | **OpenAI Whisper** (`base`) | Text transcription |
| 3 | **RoBERTa** (go_emotions) | Text-level emotion from transcription |

**Output example:**
```
🗣  Speech Emotion:  → Angry
📝 Transcription:   → "I am really tired of this situation"
💬 Text Emotion:    → Frustration (0.76), Sadness (0.42)
```

### Option 3 — Multimodal Recording
Records **face and voice simultaneously**, then analyzes all modalities:

- 🎥 **OpenFace** runs as a background process (webcam, non-blocking)
- 🎤 **Microphone** records continuously via non-blocking `sounddevice` stream
- Press **Enter** in the terminal to stop both recordings at any time
- Reuses the Voice Analysis pipeline (Option 2 logic) — no duplicate recordings

**Output example:**
```
📊 MULTIMODAL ANALYSIS REPORT
📝 Transcription:       → "..."
💬 Text Emotion:        → ...
🗣  Voice Emotion (SER): → ...
🙂 Face Emotion Timeline:
   0.00s – 3.20s : Neutral
   3.20s – 7.80s : Angry
```

### Option 4 — Live Multimodal Chat (V2 Streaming)
Runs the full **V2 Real-Time Streaming Architecture** with all three modalities active concurrently:

- **Dynamic VAD** via `faster-whisper` — detects natural sentence boundaries using trailing silence
- **Queue Broadcasting** — microphone audio simultaneously fed to STT and SER workers
- **Adaptive Emotion Fusion** — combines Face + Voice + Text with dynamic reliability weighting
- **Conflict Detection** — identifies masked emotions (e.g., `masked_anger`, `suppressed_frustration`)
- **Temporal Memory** — tracks emotional trends across the last N speaker turns
- **LLM Adapter Output** — emits structured JSON per turn, expanding 6 core emotions into a 15-class psychological model ready for generative AI pipelines

Press **Ctrl+C** to end the session.

---

## Running Individual Modules

```bash
python src/stt/record_transcribe.py          # STT only
python src/ser/record_analyze.py             # SER only
python src/faceexpression/record_express.py  # Face only
python src/streaming/live_orchestrator.py    # V2 Streaming only
python src/text_emotion/analysis.py          # Text emotion test
```
