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
const jsonPanel = document.getElementById('json-panel');
const jsonOutput = document.getElementById('json-output');

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
        if (jsonPanel) jsonPanel.style.display = 'none';
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

    const sortedProbs = Object.entries(emotion_analysis.emotion_probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

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

    // Set JSON output
    if (jsonPanel && jsonOutput) {
        jsonOutput.textContent = JSON.stringify(payload, null, 2);
        jsonPanel.style.display = 'flex';
    }

    // Scroll to results if not streaming
    if (currentTab !== 'stream') {
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }
}

// ============================================================================
// Bug 3 Fix — WAV Encoder
// ============================================================================
// Browser MediaRecorder cannot produce real PCM WAV — it always outputs WebM
// or Ogg regardless of the MIME type set on the Blob. Passing WebM bytes to
// soundfile on the server causes a hard crash in the voice pipeline.
//
// Fix: encode the raw PCM samples captured via AudioContext + ScriptProcessor
// directly into a proper WAV file (44-byte RIFF header + 16-bit PCM samples)
// entirely in the browser, with no external libraries needed.

/**
 * Encodes a Float32Array of PCM samples into a WAV Blob (16-bit mono PCM).
 * @param {Float32Array} samples  - Raw audio samples in range [-1.0, 1.0]
 * @param {number}       sampleRate - e.g. 16000 or 44100
 * @returns {Blob} - A proper audio/wav Blob that soundfile can parse
 */
function encodeWAV(samples, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataLength = samples.length * 2; // 2 bytes per 16-bit sample
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // ── RIFF header ──────────────────────────────────────────────────────────
    const writeStr = (offset, str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);       // file size - 8
    writeStr(8, 'WAVE');

    // ── fmt chunk ─────────────────────────────────────────────────────────────
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);                   // chunk size
    view.setUint16(20, 1, true);                    // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // ── data chunk ────────────────────────────────────────────────────────────
    writeStr(36, 'data');
    view.setUint32(40, dataLength, true);

    // Convert Float32 [-1, 1] → Int16 [-32768, 32767]
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 32768 : s * 32767, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
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
let audioContext = null;       // AudioContext for PCM capture
let scriptProcessor = null;    // ScriptProcessorNode
let micStream = null;          // raw MediaStream (needed to stop tracks)
let pcmSamples = [];           // accumulated Float32 samples
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

// ── Mic Record Handler (Bug 3 fix) ───────────────────────────────────────────
// Previously used MediaRecorder which produces WebM/Ogg regardless of the
// MIME type set on the Blob — soundfile on the server cannot parse it.
//
// New approach:
//   1. Open AudioContext at 16 kHz (matches SpeechBrain / Whisper expectations)
//   2. Pipe mic → ScriptProcessorNode to collect raw Float32 PCM samples
//   3. On stop: encode samples into a real WAV Blob via encodeWAV()
//   4. Server receives genuine 16-bit mono PCM WAV — soundfile parses cleanly
btnRecordVoice.addEventListener('click', async () => {
    if (isRecordingVoice) {
        // ── Stop recording ────────────────────────────────────────────────────
        isRecordingVoice = false;

        // Disconnect processor and close AudioContext
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (audioContext) {
            await audioContext.close();
            audioContext = null;
        }
        // Stop mic tracks so the browser recording indicator disappears
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }

        // Encode all collected PCM samples into a real WAV Blob
        const allSamples = new Float32Array(pcmSamples.reduce((acc, chunk) => {
            const merged = new Float32Array(acc.length + chunk.length);
            merged.set(acc);
            merged.set(chunk, acc.length);
            return merged;
        }, new Float32Array(0)));

        voiceBlob = encodeWAV(allSamples, 16000);

        btnRecordVoice.classList.remove('recording');
        btnRecordVoice.innerHTML = '<span class="record-icon">⏺</span> Record Voice';
        recordStatus.textContent = "Recording captured";
        fileNameDisplay.textContent = "browser_recording.wav";
        btnAnalyzeVoice.disabled = false;
        voiceFile.value = '';

    } else {
        // ── Start recording ───────────────────────────────────────────────────
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            pcmSamples = [];

            // Use 16000 Hz to match what SpeechBrain and Whisper expect
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            const source = audioContext.createMediaStreamSource(micStream);

            // ScriptProcessorNode collects raw PCM chunks
            // bufferSize 4096 = ~256ms at 16kHz (good balance of latency vs overhead)
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
                if (!isRecordingVoice) return;
                // Copy the buffer — the underlying memory is reused after this callback
                const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
                pcmSamples.push(chunk);
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            isRecordingVoice = true;
            btnRecordVoice.classList.add('recording');
            btnRecordVoice.innerHTML = '<span class="record-icon">⏹</span> Stop Recording';
            recordStatus.textContent = "Recording...";

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
    formData.append(
        'file',
        voiceBlob,
        fileNameDisplay.textContent === "No file chosen" ? "audio.wav" : fileNameDisplay.textContent
    );

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
                stopMultimodalSession();
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

    // ── Bug 8 fix ─────────────────────────────────────────────────────────────
    // Previously hardcoded to 127.0.0.1:8000 while API_BASE used localhost:8000.
    // Now derives host + port directly from API_BASE so both always match,
    // and the app works correctly when deployed to any host/port.
    const apiUrl   = new URL(API_BASE);
    const wsProto  = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl    = `${wsProto}//${apiUrl.host}/ws/stream`;

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