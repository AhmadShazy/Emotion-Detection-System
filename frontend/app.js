// ============================================================================
// Mode Detection — called on startup
// Fetches /health to determine which features are available
// and dynamically shows/hides tabs accordingly
// ============================================================================
const API_BASE = `${window.location.protocol}//${window.location.host}`;

let currentTab       = 'text';
let currentSessionId = null;
let availableMode    = 'text_only'; // default safe assumption

// ── Fetch mode from server ────────────────────────────────────────────────────
async function detectServerMode() {
    try {
        const res  = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        availableMode = data.mode || 'text_only';
    } catch (e) {
        console.warn('[Mode] Could not reach /health — defaulting to text_only');
        availableMode = 'text_only';
    }
    applyModeToUI();
}

// ── Apply mode to tabs ────────────────────────────────────────────────────────
function applyModeToUI() {
    const modeMap = {
        'text_only': ['text'],
        'full':      ['text', 'voice', 'multimodal', 'stream'],
    };

    const enabledTabs = modeMap[availableMode] || ['text'];

    document.querySelectorAll('.tab-btn').forEach(btn => {
        const tab = btn.getAttribute('data-tab');
        if (!enabledTabs.includes(tab)) {
            btn.classList.add('tab-disabled');
            btn.setAttribute('disabled', true);
            btn.setAttribute('title', 'Coming Soon');

            // Add coming soon badge
            if (!btn.querySelector('.coming-soon-badge')) {
                const badge = document.createElement('span');
                badge.className = 'coming-soon-badge';
                badge.textContent = 'Soon';
                btn.appendChild(badge);
            }
        } else {
            btn.classList.remove('tab-disabled');
            btn.removeAttribute('disabled');
            btn.removeAttribute('title');
        }
    });

    // If current tab got disabled, switch to text
    if (!enabledTabs.includes(currentTab)) {
        switchTab('text');
    }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tabName) {
    if (ws) disconnectWebSocket();
    if (typeof multimodalSessionId !== 'undefined' && multimodalSessionId) {
        stopMultimodalSession();
    }

    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const btn  = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
    const pane = document.getElementById(`tab-${tabName}`);

    if (btn && !btn.hasAttribute('disabled')) {
        btn.classList.add('active');
        currentTab = tabName;
    }
    if (pane) pane.classList.add('active');

    resultsPanel.style.display = 'none';
    if (jsonPanel) jsonPanel.style.display = 'none';
}

// ============================================================================
// State & Elements
// ============================================================================
const themeToggle         = document.getElementById('theme-toggle');
const htmlEl              = document.documentElement;
const tabBtns             = document.querySelectorAll('.tab-btn');
const tabPanes            = document.querySelectorAll('.tab-pane');
const loadingOverlay      = document.getElementById('loading-overlay');
const resultsPanel        = document.getElementById('results-panel');
const toastContainer      = document.getElementById('toast-container');
const jsonPanel           = document.getElementById('json-panel');
const jsonOutput          = document.getElementById('json-output');
const textInput           = document.getElementById('text-input');
const btnAnalyzeText      = document.getElementById('btn-analyze-text');
const btnRecordVoice      = document.getElementById('btn-record-voice');
const recordStatus        = document.getElementById('record-status');
const voiceFile           = document.getElementById('voice-file');
const fileNameDisplay     = document.getElementById('file-name');
const btnAnalyzeVoice     = document.getElementById('btn-analyze-voice');
const voiceWaveform       = document.getElementById('voice-waveform');
const btnStartMultimodal  = document.getElementById('btn-start-multimodal');
const btnStopMultimodal   = document.getElementById('btn-stop-multimodal');
const multimodalTimer     = document.getElementById('multimodal-timer');
const timerDisplay        = document.getElementById('timer-display');
const cameraPreview       = document.getElementById('camera-preview');
const cameraPlaceholder   = document.getElementById('camera-placeholder');
const multimodalStatus    = document.getElementById('multimodal-status');
const btnConnectStream    = document.getElementById('btn-connect-stream');
const btnDisconnectStream = document.getElementById('btn-disconnect-stream');
const streamStatusContainer = document.getElementById('stream-status-container');
const streamStatusText    = document.getElementById('stream-status-text');
const streamMicLevel      = document.getElementById('stream-mic-level');
const micLevelWrapper     = document.getElementById('mic-level-wrapper');

// ============================================================================
// Theme Management
// ============================================================================
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    htmlEl.setAttribute('data-theme', savedTheme);
}

themeToggle.addEventListener('click', () => {
    const current = htmlEl.getAttribute('data-theme');
    const next    = current === 'dark' ? 'light' : 'dark';
    htmlEl.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
});

// ============================================================================
// Tab Management
// ============================================================================
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.hasAttribute('disabled')) return;
        const tab = btn.getAttribute('data-tab');
        if (ws) disconnectWebSocket();
        if (typeof multimodalSessionId !== 'undefined' && multimodalSessionId) {
            stopMultimodalSession();
        }
        stopCameraPreview();
        stopVoiceVisualization();
        stopStreamMicMonitor();
        switchTab(tab);
    });
});

// ============================================================================
// UI Helpers
// ============================================================================
function showLoading(show, message = 'Analyzing...') {
    loadingOverlay.style.display = show ? 'flex' : 'none';
    const txt = loadingOverlay.querySelector('p');
    if (txt) txt.textContent = message;
}

function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function getEmotionColor(emotion) {
    const colors = {
        happy:       'var(--emo-happy)',
        joy:         'var(--emo-joy)',
        sad:         'var(--emo-sad)',
        angry:       'var(--emo-angry)',
        fear:        'var(--emo-fear)',
        surprised:   'var(--emo-surprised)',
        neutral:     'var(--emo-neutral)',
        disgust:     'var(--emo-disgust)',
        anxiety:     'var(--emo-anxiety)',
        frustration: 'var(--emo-angry)',
        guilt:       'var(--emo-sad)',
        shame:       'var(--emo-sad)',
        calm:        'var(--emo-neutral)',
        concerned:   'var(--emo-fear)',
        empathetic:  'var(--emo-happy)',
    };
    return colors[(emotion || '').toLowerCase()] || 'var(--primary-color)';
}

function renderResults(payload) {
    if (!payload) return;

    if (payload.session_id) currentSessionId = payload.session_id;

    const { user_input, emotion_analysis, tone_analysis } = payload;

    document.getElementById('res-text').textContent =
        user_input.text || "(No transcription)";
    document.getElementById('res-time').textContent =
        new Date(user_input.timestamp).toLocaleString();

    const emoName = document.getElementById('res-dominant-emo');
    emoName.textContent  = emotion_analysis.dominant_emotion;
    emoName.style.color  = getEmotionColor(emotion_analysis.dominant_emotion);
    document.getElementById('res-dominant-conf').textContent =
        `${Math.round(emotion_analysis.confidence * 100)}%`;

    document.getElementById('res-tone').textContent      = tone_analysis.tone;
    document.getElementById('res-tone-conf').textContent =
        `${Math.round(tone_analysis.confidence * 100)}%`;

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
                <div class="prob-bar"
                     style="width:${pct}%;background-color:${getEmotionColor(emo)}">
                </div>
            </div>
            <div class="prob-value">${pct}%</div>
        `;
        probContainer.appendChild(row);
    });

    resultsPanel.style.display = 'block';

    if (jsonPanel && jsonOutput) {
        jsonOutput.textContent = JSON.stringify(payload, null, 2);
        jsonPanel.style.display = 'flex';
    }

    if (currentTab !== 'stream') {
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }
}

// ============================================================================
// WAV Encoder
// ============================================================================
function encodeWAV(samples, sampleRate) {
    const numChannels   = 1;
    const bitsPerSample = 16;
    const byteRate      = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign    = numChannels * bitsPerSample / 8;
    const dataLength    = samples.length * 2;
    const buffer        = new ArrayBuffer(44 + dataLength);
    const view          = new DataView(buffer);

    const writeStr = (off, str) => {
        for (let i = 0; i < str.length; i++)
            view.setUint8(off + i, str.charCodeAt(i));
    };

    writeStr(0,  'RIFF');
    view.setUint32(4,  36 + dataLength, true);
    writeStr(8,  'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16,             true);
    view.setUint16(20,  1,             true);
    view.setUint16(22,  numChannels,   true);
    view.setUint32(24,  sampleRate,    true);
    view.setUint32(28,  byteRate,      true);
    view.setUint16(32,  blockAlign,    true);
    view.setUint16(34,  bitsPerSample, true);
    writeStr(36, 'data');
    view.setUint32(40, dataLength, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 32768 : s * 32767, true);
        offset += 2;
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

// ============================================================================
// Mic Level Monitor
// ============================================================================
function createMicLevelMonitor(stream, onLevel) {
    let running    = true;
    const ctx      = new (window.AudioContext || window.webkitAudioContext)();
    const source   = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);

    function tick() {
        if (!running) return;
        analyser.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        onLevel(Math.min(100, Math.round(avg * 2)));
        requestAnimationFrame(tick);
    }
    tick();

    return function stop() {
        running = false;
        try { source.disconnect(); ctx.close(); } catch (_) {}
    };
}

// ============================================================================
// Option 1: Text Analysis
// ============================================================================
btnAnalyzeText.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) { showToast("Please enter some text."); return; }

    showLoading(true, 'Analyzing text...');
    btnAnalyzeText.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/analyze/text`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        renderResults(await res.json());
    } catch (err) {
        showToast(err.message);
    } finally {
        showLoading(false);
        btnAnalyzeText.disabled = false;
    }
});

// ============================================================================
// Option 2: Voice Analysis
// ============================================================================
let voiceBlob        = null;
let voiceAudioCtx    = null;
let voiceScriptProc  = null;
let voiceMicStream   = null;
let voicePcmSamples  = [];
let isRecordingVoice = false;
let voiceAnimFrame   = null;

function startVoiceVisualization(stream) {
    if (!voiceWaveform) return;
    voiceWaveform.innerHTML = '';

    const BAR_COUNT = 20;
    const bars = [];
    for (let i = 0; i < BAR_COUNT; i++) {
        const bar = document.createElement('div');
        bar.className = 'wave-bar';
        voiceWaveform.appendChild(bar);
        bars.push(bar);
    }

    const visCtx      = new (window.AudioContext || window.webkitAudioContext)();
    const visSrc      = visCtx.createMediaStreamSource(stream);
    const visAnalyser = visCtx.createAnalyser();
    visAnalyser.fftSize = 64;
    visSrc.connect(visAnalyser);
    const data = new Uint8Array(visAnalyser.frequencyBinCount);

    function draw() {
        if (!isRecordingVoice) {
            try { visSrc.disconnect(); visCtx.close(); } catch (_) {}
            bars.forEach(b => { b.style.height = '4px'; b.style.opacity = '0.3'; });
            return;
        }
        visAnalyser.getByteFrequencyData(data);
        bars.forEach((bar, i) => {
            const val    = data[Math.floor(i * data.length / BAR_COUNT)] || 0;
            const height = Math.max(4, Math.round(val / 255 * 60));
            bar.style.height  = `${height}px`;
            bar.style.opacity = val > 10 ? '1' : '0.3';
        });
        voiceAnimFrame = requestAnimationFrame(draw);
    }
    draw();
}

function stopVoiceVisualization() {
    if (voiceAnimFrame) { cancelAnimationFrame(voiceAnimFrame); voiceAnimFrame = null; }
    if (voiceWaveform)  voiceWaveform.innerHTML = '';
}

voiceFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.wav')) {
        showToast("Please select a .wav file.");
        voiceFile.value = '';
        return;
    }
    fileNameDisplay.textContent = file.name;
    voiceBlob = file;
    btnAnalyzeVoice.disabled = false;
    recordStatus.textContent = "File selected — ready to analyze";
});

btnRecordVoice.addEventListener('click', async () => {
    if (isRecordingVoice) {
        isRecordingVoice = false;
        stopVoiceVisualization();

        if (voiceScriptProc) { voiceScriptProc.disconnect(); voiceScriptProc = null; }
        if (voiceAudioCtx)   { await voiceAudioCtx.close();  voiceAudioCtx = null; }
        if (voiceMicStream)  {
            voiceMicStream.getTracks().forEach(t => t.stop());
            voiceMicStream = null;
        }

        const merged = voicePcmSamples.reduce((acc, chunk) => {
            const out = new Float32Array(acc.length + chunk.length);
            out.set(acc); out.set(chunk, acc.length);
            return out;
        }, new Float32Array(0));

        voiceBlob = encodeWAV(merged, 16000);

        if (voiceBlob.size < 16000) {
            showToast("Recording too short — please speak for at least 1 second.");
            voiceBlob = null;
            btnRecordVoice.classList.remove('recording');
            btnRecordVoice.innerHTML = '<span class="record-icon">⏺</span> Record Voice';
            recordStatus.textContent = "Ready";
            return;
        }

        btnRecordVoice.classList.remove('recording');
        btnRecordVoice.innerHTML = '<span class="record-icon">⏺</span> Record Voice';
        recordStatus.textContent = "✅ Recording captured — click Analyze Voice";
        fileNameDisplay.textContent = "browser_recording.wav";
        btnAnalyzeVoice.disabled = false;
        voiceFile.value = '';

    } else {
        try {
            voiceMicStream  = await navigator.mediaDevices.getUserMedia({ audio: true });
            voicePcmSamples = [];

            voiceAudioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
            });

            const source    = voiceAudioCtx.createMediaStreamSource(voiceMicStream);
            voiceScriptProc = voiceAudioCtx.createScriptProcessor(4096, 1, 1);
            voiceScriptProc.onaudioprocess = (e) => {
                if (!isRecordingVoice) return;
                voicePcmSamples.push(
                    new Float32Array(e.inputBuffer.getChannelData(0))
                );
            };
            source.connect(voiceScriptProc);
            voiceScriptProc.connect(voiceAudioCtx.destination);

            isRecordingVoice = true;
            startVoiceVisualization(voiceMicStream);

            btnRecordVoice.classList.add('recording');
            btnRecordVoice.innerHTML = '<span class="record-icon">⏹</span> Stop Recording';
            recordStatus.textContent = "🔴 Recording... speak now";
            btnAnalyzeVoice.disabled = true;

        } catch (err) {
            showToast("Microphone access denied or unavailable.");
            console.error(err);
        }
    }
});

btnAnalyzeVoice.addEventListener('click', async () => {
    if (!voiceBlob) return;
    if (voiceBlob.size < 16000) {
        showToast("Recording too short. Please record at least 1 second of audio.");
        return;
    }

    recordStatus.textContent = "📤 Sending for analysis...";
    showLoading(true, 'Processing voice — this may take a few seconds...');
    btnAnalyzeVoice.disabled = true;
    btnRecordVoice.disabled  = true;

    const formData = new FormData();
    formData.append(
        'file',
        voiceBlob,
        fileNameDisplay.textContent === "No file chosen" ? "audio.wav" : fileNameDisplay.textContent
    );

    try {
        const res = await fetch(`${API_BASE}/analyze/voice`, {
            method: 'POST',
            body:   formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        recordStatus.textContent = "✅ Analysis complete";
        renderResults(await res.json());
    } catch (err) {
        recordStatus.textContent = "❌ Analysis failed";
        showToast(err.message);
    } finally {
        showLoading(false);
        btnAnalyzeVoice.disabled = false;
        btnRecordVoice.disabled  = false;
    }
});

// ============================================================================
// Option 3: Multimodal Session
// ============================================================================
let multimodalSessionId = null;
let multimodalInterval  = null;
let multimodalSeconds   = 0;
let cameraStream        = null;

async function startCameraPreview() {
    if (!cameraPreview) return;
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240, facingMode: 'user' },
            audio: false,
        });
        cameraPreview.srcObject      = cameraStream;
        cameraPreview.style.display  = 'block';
        if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
        cameraPreview.play();
        if (multimodalStatus)
            multimodalStatus.textContent = '📷 Camera active — server is recording';
    } catch (err) {
        console.warn('Camera preview unavailable:', err.message);
        if (multimodalStatus)
            multimodalStatus.textContent =
                '⚠️ Camera preview unavailable (server records independently)';
    }
}

function stopCameraPreview() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    if (cameraPreview) {
        cameraPreview.srcObject     = null;
        cameraPreview.style.display = 'none';
    }
    if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
}

async function stopMultimodalSession() {
    if (!multimodalSessionId) return;

    const sessionToStop = multimodalSessionId;
    multimodalSessionId = null;

    clearInterval(multimodalInterval);
    multimodalInterval = null;

    btnStartMultimodal.style.display = 'inline-block';
    btnStartMultimodal.disabled      = false;
    btnStopMultimodal.style.display  = 'none';
    multimodalTimer.style.display    = 'none';
    stopCameraPreview();

    if (multimodalStatus) multimodalStatus.textContent = '⏳ Processing recording...';
    showLoading(true, 'Processing multimodal recording...');

    try {
        const res = await fetch(`${API_BASE}/analyze/multimodal/stop`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ session_id: sessionToStop }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        if (multimodalStatus) multimodalStatus.textContent = '✅ Analysis complete';
        renderResults(await res.json());
    } catch (err) {
        if (multimodalStatus) multimodalStatus.textContent = `❌ ${err.message}`;
        showToast(err.message);
    } finally {
        showLoading(false);
    }
}

btnStartMultimodal.addEventListener('click', async () => {
    btnStartMultimodal.disabled = true;
    if (multimodalStatus) multimodalStatus.textContent = '🚀 Starting recording...';

    try {
        const res = await fetch(`${API_BASE}/analyze/multimodal/start`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({}),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data          = await res.json();
        multimodalSessionId = data.session_id;

        await startCameraPreview();

        btnStartMultimodal.style.display = 'none';
        btnStopMultimodal.style.display  = 'inline-block';
        multimodalTimer.style.display    = 'flex';

        multimodalSeconds        = 0;
        timerDisplay.textContent = "00:00";

        multimodalInterval = setInterval(() => {
            multimodalSeconds++;
            const mm = String(Math.floor(multimodalSeconds / 60)).padStart(2, '0');
            const ss = String(multimodalSeconds % 60).padStart(2, '0');
            timerDisplay.textContent = `${mm}:${ss}`;
            if (multimodalSeconds >= 30) stopMultimodalSession();
        }, 1000);

    } catch (err) {
        btnStartMultimodal.disabled      = false;
        btnStartMultimodal.style.display = 'inline-block';
        if (multimodalStatus) multimodalStatus.textContent = `❌ ${err.message}`;
        showToast(err.message);
    }
});

btnStopMultimodal.addEventListener('click', stopMultimodalSession);

// ============================================================================
// Option 4: Live Stream (WebSocket)
// ============================================================================
let ws              = null;
let streamMicStream = null;
let streamMicStopFn = null;

function stopStreamMicMonitor() {
    if (streamMicStopFn) { streamMicStopFn(); streamMicStopFn = null; }
    if (streamMicStream) {
        streamMicStream.getTracks().forEach(t => t.stop());
        streamMicStream = null;
    }
    if (streamMicLevel)  streamMicLevel.style.width = '0%';
    if (micLevelWrapper) micLevelWrapper.style.display = 'none';
}

function disconnectWebSocket() {
    if (ws) { ws.close(); ws = null; }
    stopStreamMicMonitor();
    showLoading(false);
    btnConnectStream.style.display      = 'inline-block';
    btnConnectStream.disabled           = false;
    btnDisconnectStream.style.display   = 'none';
    streamStatusContainer.style.display = 'none';
}

btnConnectStream.addEventListener('click', async () => {
    btnConnectStream.disabled = true;
    streamStatusContainer.style.display = 'flex';
    streamStatusText.textContent = "Connecting...";

    try {
        streamMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (streamMicLevel && micLevelWrapper) {
            micLevelWrapper.style.display = 'flex';
            streamMicStopFn = createMicLevelMonitor(streamMicStream, (level) => {
                streamMicLevel.style.width = `${level}%`;
            });
        }
    } catch (e) {
        console.warn('Stream mic monitor unavailable:', e.message);
    }

    const apiUrl      = new URL(API_BASE);
    const wsProto     = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const sessionParam = currentSessionId
        ? `&session_id=${encodeURIComponent(currentSessionId)}` : '';
    const wsUrl       = `${wsProto}//${apiUrl.host}/ws/stream?_=${Date.now()}${sessionParam}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        btnConnectStream.style.display    = 'none';
        btnDisconnectStream.style.display = 'inline-block';
        btnConnectStream.disabled         = false;
        streamStatusText.textContent      = "🔴 Connected — speak now...";
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'error') {
                showToast(data.message || 'Streaming error occurred.', 'error');
                disconnectWebSocket();
                return;
            }
            if (data.type === 'status') {
                streamStatusText.textContent = data.message || "Connected";
                return;
            }
            if (data.session_id) {
                renderResults(data);
                streamStatusText.textContent = "✅ Result received — speak again...";
                return;
            }
        } catch (e) {
            console.error("WS parse error:", e);
        }
    };

    ws.onerror = () => {
        showToast("WebSocket connection error.");
        disconnectWebSocket();
    };

    ws.onclose = () => { disconnectWebSocket(); };
});

btnDisconnectStream.addEventListener('click', disconnectWebSocket);

// ============================================================================
// Init
// ============================================================================
initTheme();
detectServerMode();