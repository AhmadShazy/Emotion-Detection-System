// ============================================================================
// State & Elements
// ============================================================================
const API_BASE = 'http://localhost:8000';
let currentTab = 'text';

// Theme
const themeToggle = document.getElementById('theme-toggle');
const htmlEl = document.documentElement;

// Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

// Overlay & Results
const loadingOverlay = document.getElementById('loading-overlay');
const resultsPanel = document.getElementById('results-panel');
const toastContainer = document.getElementById('toast-container');

// Text
const textInput = document.getElementById('text-input');
const btnAnalyzeText = document.getElementById('btn-analyze-text');

// Voice
const btnRecordVoice = document.getElementById('btn-record-voice');
const recordStatus = document.getElementById('record-status');
const voiceFile = document.getElementById('voice-file');
const fileNameDisplay = document.getElementById('file-name');
const btnAnalyzeVoice = document.getElementById('btn-analyze-voice');

// Multimodal
const btnStartMultimodal = document.getElementById('btn-start-multimodal');
const btnStopMultimodal = document.getElementById('btn-stop-multimodal');
const multimodalTimer = document.getElementById('multimodal-timer');
const timerDisplay = document.getElementById('timer-display');

// Stream
const btnConnectStream = document.getElementById('btn-connect-stream');
const btnDisconnectStream = document.getElementById('btn-disconnect-stream');
const streamStatusContainer = document.getElementById('stream-status-container');
const streamStatusText = document.getElementById('stream-status-text');

// ============================================================================
// Theme Management
// ============================================================================
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    htmlEl.setAttribute('data-theme', savedTheme);
}

themeToggle.addEventListener('click', () => {
    const current = htmlEl.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    htmlEl.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});

// ============================================================================
// Tab Management
// ============================================================================
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Reset states if leaving active session
        if (ws) disconnectWebSocket();
        if (multimodalSessionId) stopMultimodalSession();

        // UI update
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        currentTab = btn.getAttribute('data-tab');
        document.getElementById(`tab-${currentTab}`).classList.add('active');
        
        // Hide results when switching tabs
        resultsPanel.style.display = 'none';
    });
});

// ============================================================================
// UI Helpers
// ============================================================================
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function getEmotionColor(emotion) {
    const colors = {
        happy: 'var(--emo-happy)',
        joy: 'var(--emo-joy)',
        sad: 'var(--emo-sad)',
        angry: 'var(--emo-angry)',
        fear: 'var(--emo-fear)',
        surprised: 'var(--emo-surprised)',
        neutral: 'var(--emo-neutral)',
        disgust: 'var(--emo-disgust)',
        anxiety: 'var(--emo-anxiety)',
    };
    return colors[emotion.toLowerCase()] || 'var(--primary-color)';
}

function renderResults(payload) {
    if (!payload) return;
    
    // Unpack payload
    const { user_input, emotion_analysis, tone_analysis } = payload;
    
    // Header
    document.getElementById('res-text').textContent = user_input.text || "(No text)";
    document.getElementById('res-time').textContent = new Date(user_input.timestamp).toLocaleString();
    
    // Dominant Emotion
    const emoName = document.getElementById('res-dominant-emo');
    emoName.textContent = emotion_analysis.dominant_emotion;
    emoName.style.color = getEmotionColor(emotion_analysis.dominant_emotion);
    document.getElementById('res-dominant-conf').textContent = `${Math.round(emotion_analysis.confidence * 100)}%`;
    
    // Tone
    document.getElementById('res-tone').textContent = tone_analysis.tone;
    document.getElementById('res-tone-conf').textContent = `${Math.round(tone_analysis.confidence * 100)}%`;
    
    // Probabilities
    const probContainer = document.getElementById('prob-bars');
    probContainer.innerHTML = '';
    
    // Sort probabilities descending
    const sortedProbs = Object.entries(emotion_analysis.emotion_probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Show top 5
        
    sortedProbs.forEach(([emo, val]) => {
        const pct = Math.round(val * 100);
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-label">${emo}</div>
            <div class="prob-bar-wrapper">
                <div class="prob-bar" style="width: ${pct}%; background-color: ${getEmotionColor(emo)}"></div>
            </div>
            <div class="prob-value">${pct}%</div>
        `;
        probContainer.appendChild(row);
    });
    
    resultsPanel.style.display = 'block';
    
    // Scroll to results if not streaming
    if (currentTab !== 'stream') {
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }
}

// ============================================================================
// Option 1: Text Analysis
// ============================================================================
btnAnalyzeText.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
        showToast("Please enter some text.");
        return;
    }
    
    showLoading(true);
    btnAnalyzeText.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/analyze/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        showToast(err.message);
    } finally {
        showLoading(false);
        btnAnalyzeText.disabled = false;
    }
});

// ============================================================================
// Option 2: Voice Analysis (Upload + Record)
// ============================================================================
let voiceBlob = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecordingVoice = false;

// File Upload Handler
voiceFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        if (!file.name.toLowerCase().endsWith('.wav')) {
            showToast("Please select a .wav file.");
            voiceFile.value = '';
            return;
        }
        fileNameDisplay.textContent = file.name;
        voiceBlob = file;
        btnAnalyzeVoice.disabled = false;
        recordStatus.textContent = "File selected";
    }
});

// Mic Record Handler
btnRecordVoice.addEventListener('click', async () => {
    if (isRecordingVoice) {
        // Stop recording
        mediaRecorder.stop();
        btnRecordVoice.classList.remove('recording');
        btnRecordVoice.innerHTML = '<span class="record-icon">⏺</span> Record Voice';
        recordStatus.textContent = "Processing...";
        isRecordingVoice = false;
    } else {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            
            // Try to force standard webm or wav if browser supports it
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
            mediaRecorder = new MediaRecorder(stream, { mimeType });
            
            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };
            
            mediaRecorder.onstop = () => {
                // Note: Browser MediaRecorder doesn't easily create pure PCM .wav.
                // We create a blob and the server's pipeline will handle conversion if needed
                // (Though ideally the server expects .wav. We'll send it as .wav and hope soundfile/ffmpeg handles it).
                // A robust solution uses an audio worklet to encode PCM WAV in browser.
                voiceBlob = new Blob(audioChunks, { type: 'audio/wav' }); 
                
                recordStatus.textContent = "Recording captured";
                fileNameDisplay.textContent = "browser_recording.wav";
                btnAnalyzeVoice.disabled = false;
                
                // Stop tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            isRecordingVoice = true;
            btnRecordVoice.classList.add('recording');
            btnRecordVoice.innerHTML = '<span class="record-icon">⏹</span> Stop Recording';
            recordStatus.textContent = "Recording...";
            
            // Reset file input
            voiceFile.value = '';
        } catch (err) {
            showToast("Microphone access denied or unavailable.");
            console.error(err);
        }
    }
});

// Analyze Voice
btnAnalyzeVoice.addEventListener('click', async () => {
    if (!voiceBlob) return;
    
    showLoading(true);
    btnAnalyzeVoice.disabled = true;
    
    const formData = new FormData();
    formData.append('file', voiceBlob, fileNameDisplay.textContent === "No file chosen" ? "audio.wav" : fileNameDisplay.textContent);
    
    try {
        const res = await fetch(`${API_BASE}/analyze/voice`, {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        showToast(err.message);
    } finally {
        showLoading(false);
        btnAnalyzeVoice.disabled = false;
    }
});


// ============================================================================
// Option 3: Multimodal Session
// ============================================================================
let multimodalSessionId = null;
let multimodalInterval = null;
let multimodalSeconds = 0;

async function stopMultimodalSession() {
    if (!multimodalSessionId) return;
    
    clearInterval(multimodalInterval);
    btnStartMultimodal.style.display = 'inline-block';
    btnStopMultimodal.style.display = 'none';
    multimodalTimer.style.display = 'none';
    
    showLoading(true);
    
    try {
        const res = await fetch(`${API_BASE}/analyze/multimodal/stop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: multimodalSessionId })
        });
        
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        renderResults(data);
    } catch (err) {
        showToast(err.message);
    } finally {
        showLoading(false);
        multimodalSessionId = null;
    }
}

btnStartMultimodal.addEventListener('click', async () => {
    btnStartMultimodal.disabled = true;
    try {
        const res = await fetch(`${API_BASE}/analyze/multimodal/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        
        multimodalSessionId = data.session_id;
        
        // Update UI
        btnStartMultimodal.style.display = 'none';
        btnStopMultimodal.style.display = 'inline-block';
        multimodalTimer.style.display = 'flex';
        
        // Start Timer
        multimodalSeconds = 0;
        timerDisplay.textContent = "00:00";
        multimodalInterval = setInterval(() => {
            multimodalSeconds++;
            const s = String(multimodalSeconds).padStart(2, '0');
            timerDisplay.textContent = `00:${s}`;
            
            if (multimodalSeconds >= 30) {
                stopMultimodalSession(); // Auto stop at hard cap
            }
        }, 1000);
        
    } catch (err) {
        showToast(err.message);
    } finally {
        btnStartMultimodal.disabled = false;
    }
});

btnStopMultimodal.addEventListener('click', stopMultimodalSession);


// ============================================================================
// Option 4: Live Stream (WebSocket)
// ============================================================================
let ws = null;

function disconnectWebSocket() {
    if (ws) {
        ws.close();
        ws = null;
    }
    btnConnectStream.style.display = 'inline-block';
    btnDisconnectStream.style.display = 'none';
    streamStatusContainer.style.display = 'none';
}

btnConnectStream.addEventListener('click', () => {
    btnConnectStream.disabled = true;
    streamStatusContainer.style.display = 'flex';
    streamStatusText.textContent = "Connecting...";
    
    // Determine WS protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//127.0.0.1:8000/ws/stream`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        btnConnectStream.style.display = 'none';
        btnDisconnectStream.style.display = 'inline-block';
        btnConnectStream.disabled = false;
        streamStatusText.textContent = "Connected. Speak now...";
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'status') {
                streamStatusText.textContent = data.message;
            } else if (data.session_id) {
                // It's a unified emotion payload
                renderResults(data);
                streamStatusText.textContent = "Analyzing... Speak again.";
            }
        } catch (e) {
            console.error("WS Parse error", e);
        }
    };
    
    ws.onerror = (error) => {
        console.error("WS Error:", error);
        showToast("WebSocket connection error.");
        disconnectWebSocket();
    };
    
    ws.onclose = () => {
        disconnectWebSocket();
    };
});

btnDisconnectStream.addEventListener('click', disconnectWebSocket);

// ============================================================================
// Init
// ============================================================================
initTheme();
