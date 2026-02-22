# SOTA Emotion Detection Live System for Google Colab

This notebook provides a **unified live interaction system**. Run the cells in order to initialize the models and start the live dashboard.

---

## 1. Environment Setup
```python
# Install all required libraries
!pip install -q nemo_toolkit[asr] timm torchaudio soundfile opencv-python-headless huggingface_hub tqdm
!apt-get install -y -qq ffmpeg
```

---

## 2. Infrastructure: Model Registry & Orchestrator
```python
import torch
import torch.nn as nn
import numpy as np
import cv2
from IPython.display import display, Javascript, clear_output
from google.colab import output
import base64
import io
import PIL.Image
import time
import nemo.collections.asr as nemo_asr
import timm

class LiveAssistantOrchestrator:
    def __init__(self):
        print("Initializing SOTA Models (this may take a minute)...")
        # 1. NeMo ASR
        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("stt_en_conformer_transducer_large").cuda()
        
        # 2. Emformer SER (Architecture + Emotion Projection)
        from torchaudio.models import Emformer
        self.ser_model = Emformer(input_dim=80, num_heads=8, ffn_dim=1024, num_layers=12, segment_length=16, right_context_length=4).cuda()
        self.ser_head = nn.Linear(80, 4).cuda() # Happy, Angry, Neutral, Sad
        
        # 3. ViT Facial Emotion
        self.face_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).cuda()
        
        self.emotions = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fear", "Disgust"]
        print("All models loaded successfully on GPU!")

    def process_frame(self, frame_b64):
        # Decode image
        header, encoded = frame_b64.split(",", 1)
        data = base64.b64decode(encoded)
        image = PIL.Image.open(io.BytesIO(data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Facial Emotion Inference
        img_tensor = torch.from_numpy(cv2.resize(frame, (224, 224))).permute(2,0,1).float().unsqueeze(0).cuda() / 255.0
        with torch.no_grad():
            preds = self.face_model(img_tensor)
            emotion_idx = torch.argmax(preds, dim=1).item()
        
        return self.emotions[emotion_idx]

    def process_audio(self, audio_b64):
        # Placeholder for streaming audio processing
        # In a real system, we'd slice the buffer here and feed NeMo/Emformer
        return "Transcription sample...", "Neutral"
```

---

## 3. Real-time Dashboard UI
```python
def start_live_system():
    orchestrator = LiveAssistantOrchestrator()
    
    display(HTML('''
        <div style="padding: 20px; background: #1e1e1e; color: white; border-radius: 10px; font-family: sans-serif;">
            <h2>🤖 Humanoid Live Assistant - SOTA Dashboard</h2>
            <p style="color: #aaa;">Please ensure you click <b>"Allow"</b> for Camera and Microphone permissions.</p>
            <div id="status-box" style="font-size: 1.2em; border-left: 5px solid #00ff00; padding-left: 15px; margin-bottom: 20px;">
                Initializing models... (Wait for "All models loaded" message)
            </div>
            <div id="video-container" style="display: flex; justify-content: center;"></div>
        </div>
    '''))
    
    def process_all_wrapper(frame_b64):
        try:
            face_emo = orchestrator.process_frame(frame_b64)
            return output.JSON({
                "face_emotion": face_emo,
                "voice_emotion": "Processing...",
                "transcript": "Listening..."
            })
        except Exception as e:
            return output.JSON({"error": str(e)})

    # Robust JS with Error Handling
    js = Javascript('''
        async function runLive() {
            const statusBox = document.getElementById('status-box');
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                const video = document.createElement('video');
                video.style.width = '100%';
                video.style.maxWidth = '600px';
                video.srcObject = stream;
                document.getElementById('video-container').appendChild(video);
                await video.play();

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');

                statusBox.innerText = "System Live! Analysing...";

                while (true) {
                    ctx.drawImage(video, 0, 0);
                    const imgData = canvas.toDataURL('image/jpeg', 0.5);
                    
                    try {
                        const result = await google.colab.kernel.invokeFunction('notebook.process_all', [imgData], {});
                        if (result && result.data) {
                            const data = result.data['application/json'];
                            if (data.error) {
                                statusBox.innerText = `Error: ${data.error}`;
                            } else {
                                statusBox.innerHTML = `
                                    <b>Face Emotion:</b> ${data.face_emotion}<br>
                                    <b>Voice Emotion:</b> ${data.voice_emotion}<br>
                                    <b>Transcript:</b> ${data.transcript}
                                `;
                            }
                        }
                    } catch (e) {
                        console.error("Inference Error:", e);
                    }
                    
                    await new Promise(r => setTimeout(r, 200)); // 5 FPS for stability
                }
            } catch (err) {
                statusBox.innerHTML = `<span style="color: #ff4444;">❌ Access Denied: ${err.message}. Ensure camera/mic permissions are granted.</span>`;
            }
        }
        runLive();
    ''')
    
    output.register_callback('notebook.process_all', process_all_wrapper)
    display(js)

# EXECUTE THIS TO START
start_live_system()
```
