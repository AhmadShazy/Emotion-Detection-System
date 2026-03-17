# Humanoid Assistant Demo

This project demonstrates a modular Humanoid Assistant capable of Speech-to-Text (STT), Speech Emotion Recognition (SER), and Facial Expression Analysis.

## Project Structure

- **main.py**: The central entry point for the application. Run this to access the main menu.
- **src/**: Source code for the project modules.
  - **stt/**: Speech-to-Text module using OpenAI's **Whisper**.
  - **ser/**: Speech Emotion Recognition module using **SpeechBrain** (Wav2Vec2-IEMOCAP).
  - **faceexpression/**: Face Expression Analysis utilizing external **OpenFace** binaries.
  - **full_analysis/**: Orchestrates all three modules for a complete interaction analysis.
  - **text_emotion/**: Helper module for analyzing emotion from transcribed text.
- **external/**:
  - **openface/**: Directory for the OpenFace executable (required for face analysis).
  - **ser_project/**: Contains requirements for the SER module.
- **data/**:
  - **recordings/**: Stores audio recordings from the microphone.
  - **processed/**: Stores intermediate processing results (e.g., OpenFace CSVs).

## Setup & Requirements

1.  **Python Environment**: Ensure you are running in the project's virtual environment.
2.  **Dependencies**: Install python dependencies (see `requirements.txt` in root or `external/ser_project/`).
    *   Key libraries: `speechbrain`, `transformers`, `torch`, `whisper`, `sounddevice`, `numpy`.
3.  **OpenFace**: Ensure the OpenFace executable is correctly placed in `external/openface/OpenFace_2.2.0_win_x64/FeatureExtraction.exe`.

## Usage

### Main Menu
The easiest way to use the tools is via the main menu:

```bash
python main.py
```

This will present options to run:
1.  **Face Expression Analysis** (Live webcam recording via OpenFace)
2.  **Voice Emotion Analysis** (Audio recording + SpeechBrain inference)
3.  **Speech-to-Text** (Audio recording + Whisper transcription)
4.  **Full Analysis** (Runs all pipelines sequentially/together)
5.  **V2 Real-Time Streaming Architecture [Live]** (Runs continuous, concurrent evaluation of Face, Voice, and Text modalities, driven by an intelligent dynamic Voice Activity Detector).

### V2 Streaming Architecture Features
- **Queue Broadcasting:** Microphone audio is simultaneously broadcast to independent worker threads without starvation.
- **Dynamic VAD:** Faster-Whisper dynamically groups speech into natural sentences by tracking 1.5-second trailing silences, allowing unconstrained turn phrasing.
- **Anti-Hallucination & Anti-Speaking Fillers:** VAD actively discards non-speech mic bumps, and OpenFace utilizes an active penalty mask to ensure mouth articulation isn't falsely classified as smiling (`Happy`).
- **Adaptive Emotion Fusion:** The Orchestrator resolves conflicts between Face, Voice, and Text by calculating dynamic reliability scores (based on volume, frame stability, and semantics) instead of static averages. It actively detects hidden tone shifts (e.g., Masked Anger).
- **Temporal Memory:** The system tracks the last $N$ speaker turns to evaluate emotional stability, recognizing gradual mood trends and overall volatility rather than treating each sentence in isolation.
- **Rich Synchronous Output:** Emits a heavily structured JSON payload representing a complete "Turn Record" combining Face, Voice, Text, conflict resolution, and conversation history context in under ~0.5 seconds per turn.

### Individual Modules

You can also run modules individually if their scripts allow (e.g., for testing):

- **STT**: `python src/stt/record_transcribe.py`
- **SER**: `python src/ser/record_analyze.py`
- **Face**: `python src/faceexpression/record_express.py`
