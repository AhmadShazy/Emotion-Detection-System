# Humanoid Assistant V2

<div align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Vanilla_JS-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="Vanilla JS">
</div>

<br/>

**Humanoid Assistant V2** is a modular, real-time, multimodal emotion-analysis system. It captures user input and analyzes psychological states across three concurrent modalities: **Text**, **Voice**, and **Facial Expressions**. Using advanced deep learning models and a custom fusion engine, it provides a comprehensive emotional profile ready for integration with generative AI pipelines.

---

## ✨ Key Features

- **Multimodal Emotion Fusion**: Intelligently combines textual semantics, vocal tone, and facial action units (AUs) to detect complex states like *masked anger* or *suppressed frustration*.
- **Real-Time Streaming Architecture**: Built on WebSockets and `faster-whisper` VAD to process live audio and video feeds without blocking.
- **Thread-Safe Session Isolation**: Safely handles concurrent requests from multiple clients with independent temporal emotion memory and state managers.
- **Modern Web Interface**: A sleek, responsive Vanilla JS frontend that handles 16kHz PCM audio capturing and live WebSocket telemetry.
- **State-of-the-Art ML Stack**:
  - **Speech-to-Text (STT)**: OpenAI Whisper & `faster-whisper`.
  - **Speech Emotion Recognition (SER)**: SpeechBrain (Wav2Vec2 IEMOCAP).
  - **Text Emotion**: RoBERTa (`SamLowe/roberta-base-go_emotions`).
  - **Facial Expression**: OpenFace 2.2.0 + Custom AU Classifier.

---

## 🏗️ System Architecture

```text
humanoid-assistant-demo/
├── api.py                          # FastAPI application entry point
├── main.py                         # Legacy CLI entry point
├── requirements.txt                # Python dependencies
├── frontend/                       # Modern Web UI assets
│   ├── index.html                  # Main UI layout
│   ├── app.js                      # UI logic, audio capture, WebSocket
│   └── index.css                   # Theming and styling
├── routers/                        # Modular FastAPI endpoint routes
│   ├── text.py                     # Text emotion API
│   ├── voice.py                    # Voice emotion API
│   ├── multimodal.py               # Multimodal analysis API
│   └── stream.py                   # WebSocket streaming API
├── schemas/                        # Pydantic data models
│   └── emotion.py                  # Standardized JSON response schemas
├── external/
│   ├── openface/                   # OpenFace binary folder (Required)
│   └── whisper/                    # Whisper model cache
├── data/                           # Runtime data (WAVs, CSVs)
└── src/                            # Core Analysis Logic
    ├── faceexpression/             # OpenFace pipeline & AU classification
    ├── ser/                        # SpeechBrain Wav2Vec2 SER engine
    ├── stt/                        # Whisper speech-to-text
    ├── text_emotion/               # RoBERTa text emotion
    └── streaming/                  # Thread-safe session managers & pipelines
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3.9+** (Tested on Windows).
- **Git** (for cloning).
- **Microphone and Webcam** (for multimodal and live-stream features).

### 2. Environment Setup
Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/AhmadShazy/Emotion-Detection-System.git
cd Emotion-Detection-System
python -m venv .venv
# Activate the virtual environment:
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. OpenFace Integration
Facial expression analysis requires the OpenFace C++ binary.
1. Download [OpenFace 2.2.0 (Windows x64)](https://github.com/TadasBaltrusaitis/OpenFace/releases).
2. Extract it into the `external` directory so the executable is located exactly at:
   `external/openface/OpenFace_2.2.0_win_x64/FeatureExtraction.exe`

*(Note: Hugging Face models for Whisper, Wav2Vec2, and RoBERTa will download automatically on their first execution.)*

---

## 🚀 Usage

The system exposes a robust REST and WebSocket API via FastAPI, served alongside a modern web application.

### Starting the Web Server

Run the following command from the root of the project:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Once the server has started, open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

### Operational Modes (Web UI)
- **💬 Text Analysis**: Paste or type text. Analyzes semantics instantly using RoBERTa.
- **🎤 Voice Analysis**: Record your voice via the browser. Automatically transcribes text via Whisper and analyzes both vocal tone (Wav2Vec2) and textual emotion.
- **🎬 Multimodal Session**: Records both your webcam (analyzed in the background via OpenFace) and microphone. Returns a fused temporal report.
- **🌐 Live Stream**: Connects via WebSockets. Captures audio continuously, uses VAD to chunk sentences, and provides real-time multimodal feedback.

*(Legacy Terminal Interface: You can still run the old terminal menu by executing `python main.py`)*

---

## 🔌 API Reference (Brief)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze/text` | Analyzes emotion from a raw string. |
| `POST` | `/analyze/voice` | Analyzes a 16kHz PCM WAV audio file. |
| `POST` | `/analyze/multimodal/start` | Initializes a background multimodal session. |
| `POST` | `/analyze/multimodal/stop` | Terminates and processes the multimodal session. |
| `WS` | `/ws/stream` | Establishes a live bidirectional analysis socket. |
| `GET` | `/health` | Returns server health and API version. |

---

## 📝 License & Authors
Developed by **Ahmad (Shezi)**. 
Designed for research and demonstration purposes in Multimodal Human-Computer Interaction.
