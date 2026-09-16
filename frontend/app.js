// ============================================================================
// API Key Gate — validates key against server before showing the UI
// Key stored in sessionStorage (cleared on browser close)
// Never hardcoded — user enters it manually
// ============================================================================
const API_BASE = `${window.location.protocol}//${window.location.host}`;
let SESSION_API_KEY = sessionStorage.getItem('humanoid_api_key') || '';

// ── Gate elements ─────────────────────────────────────────────────────────────
const apiGate       = document.getElementById('api-gate');
const apiKeyInput   = document.getElementById('api-key-input');
const apiGateSubmit = document.getElementById('api-gate-submit');
const apiGateError  = document.getElementById('api-gate-error');
const apiGateToggle = document.getElementById('api-gate-toggle');

// ── Show/hide password toggle ─────────────────────────────────────────────────
apiGateToggle.addEventListener('click', () => {
    const isPassword = apiKeyInput.type === 'password';
    apiKeyInput.type = isPassword ? 'text' : 'password';
    apiGateToggle.textContent = isPassword ? '🙈' : '👁';
});

// ── Submit on Enter key ───────────────────────────────────────────────────────
apiKeyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') apiGateSubmit.click();
});

// ── Validate key against server ───────────────────────────────────────────────
async function validateAndUnlock(key) {
    apiGateSubmit.disabled = true;
    apiGateSubmit.classList.add('loading');
    apiGateSubmit.textContent = 'Verifying...';
    apiGateError.style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/analyze/text`, {
            method:  'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': key,
            },
            body: JSON.stringify({ text: 'ping' }),
        });

        if (res.status === 403) {
            apiGateError.textContent = '❌ Invalid API key. Please try again.';
            apiGateError.style.display = 'block';
            apiGateSubmit.disabled = false;
            apiGateSubmit.classList.remove('loading');
            apiGateSubmit.textContent = 'Access System';
            sessionStorage.removeItem('humanoid_api_key');
            return;
        }

        // Key valid — store and unlock
        SESSION_API_KEY = key;
        sessionStorage.setItem('humanoid_api_key', key);
        apiGate.style.display = 'none';

    } catch (err) {
        apiGateError.textContent = '⚠️ Could not reach server. Try again.';
        apiGateError.style.display = 'block';
        apiGateSubmit.disabled = false;
        apiGateSubmit.classList.remove('loading');
        apiGateSubmit.textContent = 'Access System';
    }
}

apiGateSubmit.addEventListener('click', () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
        apiGateError.textContent = '❌ Please enter your API key.';
        apiGateError.style.display = 'block';
        return;
    }
    validateAndUnlock(key);
});

// ── Auto-unlock if valid key already in sessionStorage ────────────────────────
if (SESSION_API_KEY) {
    validateAndUnlock(SESSION_API_KEY);
} else {
    apiGate.style.display = 'flex';
}

// ============================================================================
// Session state
// ============================================================================
// There used to be mode detection here: the page asked /health which mode the
// server was in and greyed out the tabs the server could not serve, because a
// reduced text-only deployment existed. That deployment is gone and every mode
// is always available, so the tabs no longer need gating — and the page no
// longer has to guess what it is talking to.
let currentTab       = 'text';
let currentSessionId = null;

function switchTab(tabName) {
    if (ws) disconnectWebSocket();
    // Leaving the tab mid-recording must release the camera and microphone,
    // or the device stays lit and held after the user has moved on.
    if (typeof isRecordingVideo !== 'undefined' && isRecordingVideo) {
        stopVideoRecording();
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
const btnRecordVideo      = document.getElementById('btn-record-video');
const videoFile           = document.getElementById('video-file');
const videoFileName       = document.getElementById('video-file-name');
const btnAnalyzeVideo     = document.getElementById('btn-analyze-video');
const multimodalTimer     = document.getElementById('multimodal-timer');
const timerDisplay        = document.getElementById('timer-display');
const cameraPreview       = document.getElementById('camera-preview');
const cameraPlaceholder   = document.getElementById('camera-placeholder');
// The Live Call tab has its own preview elements. It must NOT reuse the ones
// above: those live inside the Multimodal pane, which is hidden while the Live
// tab is showing, so the video would render into something invisible.
const streamCameraPreview     = document.getElementById('stream-camera-preview');
const streamCameraPlaceholder = document.getElementById('stream-camera-placeholder');
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
        if (typeof isRecordingVideo !== 'undefined' && isRecordingVideo) {
            stopVideoRecording();
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
        const avg = data.reduce((a, b) => a + b,0) / data.length;
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
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': SESSION_API_KEY,
            },
            // Return the session id so the server keeps this conversation's
            // emotion memory. Without it every request minted a fresh session,
            // which pinned confidence at 0.7x raw and disabled all smoothing.
            body: JSON.stringify({ text, session_id: currentSessionId }),
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
    // Same reason as the text route — keep the server's emotion memory alive
    // across turns instead of starting a new session on every request.
    if (currentSessionId) formData.append('session_id', currentSessionId);

    try {
        const res = await fetch(`${API_BASE}/analyze/voice`, {
            method: 'POST',
            headers: { 'X-API-Key': SESSION_API_KEY },
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
// Option 3: Video Analysis
// ============================================================================
// The browser captures the video and uploads it. Previously the SERVER opened
// its own webcam and this tab only showed a decorative preview — which meant
// only one person, sitting at the server, could ever use it.
let videoBlob        = null;
let videoRecorder    = null;
let videoStream      = null;
let videoChunks      = [];
let isRecordingVideo = false;
let videoInterval    = null;
let videoSeconds     = 0;

const MAX_VIDEO_SECONDS = 30;

// Formats vary by browser: Chrome and Firefox produce WebM, Safari below 18.4
// produces MP4 only. Pick whichever this browser actually supports.
function pickVideoMimeType() {
    const candidates = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm',
        'video/mp4',
    ];
    for (const type of candidates) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
    }
    return null;
}

function setVideoStatus(text) {
    if (multimodalStatus) multimodalStatus.textContent = text;
}

function stopCameraPreview() {
    if (videoStream) {
        videoStream.getTracks().forEach(t => t.stop());
        videoStream = null;
    }
    if (cameraPreview) {
        cameraPreview.srcObject     = null;
        cameraPreview.style.display = 'none';
    }
    if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
}

function resetVideoRecordingUI() {
    isRecordingVideo = false;
    clearInterval(videoInterval);
    videoInterval = null;
    if (multimodalTimer) multimodalTimer.style.display = 'none';
    if (btnRecordVideo) {
        btnRecordVideo.classList.remove('recording');
        btnRecordVideo.innerHTML = '<span class="record-icon">⏺</span> Record Video';
    }
    stopCameraPreview();
}

async function startVideoRecording() {
    const mimeType = pickVideoMimeType();
    if (!mimeType) {
        showToast('This browser cannot record video. Choose a file instead.');
        return;
    }

    try {
        // audio: true is the whole difference from the old preview-only code —
        // without it there is no voice to analyse.
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: true,
        });
    } catch (err) {
        showToast('Camera or microphone access was denied.');
        return;
    }

    if (cameraPreview) {
        cameraPreview.srcObject     = videoStream;
        cameraPreview.style.display = 'block';
        if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
        cameraPreview.play();
    }

    videoChunks   = [];
    videoRecorder = new MediaRecorder(videoStream, { mimeType });

    videoRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) videoChunks.push(e.data);
    };

    videoRecorder.onstop = () => {
        videoBlob = new Blob(videoChunks, { type: mimeType });
        resetVideoRecordingUI();

        if (videoBlob.size < 1024) {
            showToast('That recording was too short. Try again.');
            videoBlob = null;
            setVideoStatus('Ready');
            return;
        }

        const kb = Math.round(videoBlob.size / 1024);
        setVideoStatus(`✅ Recorded ${kb} KB — click Analyze Video`);
        // Include a real extension. The server accepts either a known MIME type
        // or a known suffix, and a name with no extension fails the second check.
        const ext = mimeType.startsWith('video/mp4') ? 'mp4' : 'webm';
        if (videoFileName) videoFileName.textContent = `browser_recording.${ext}`;
        if (btnAnalyzeVideo) btnAnalyzeVideo.disabled = false;
        if (videoFile) videoFile.value = '';
    };

    videoRecorder.start();
    isRecordingVideo = true;

    btnRecordVideo.classList.add('recording');
    btnRecordVideo.innerHTML = '<span class="record-icon">⏹</span> Stop Recording';
    setVideoStatus('🔴 Recording — speak and look at the camera');
    if (btnAnalyzeVideo) btnAnalyzeVideo.disabled = true;

    videoSeconds = 0;
    if (multimodalTimer) multimodalTimer.style.display = 'flex';
    if (timerDisplay) timerDisplay.textContent = '00:00';

    videoInterval = setInterval(() => {
        videoSeconds++;
        const mm = String(Math.floor(videoSeconds / 60)).padStart(2, '0');
        const ss = String(videoSeconds % 60).padStart(2, '0');
        if (timerDisplay) timerDisplay.textContent = `${mm}:${ss}`;
        // The server analyses only the first 30 seconds, so stop there rather
        // than uploading footage that will be discarded.
        if (videoSeconds >= MAX_VIDEO_SECONDS) stopVideoRecording();
    }, 1000);
}

function stopVideoRecording() {
    if (videoRecorder && videoRecorder.state !== 'inactive') {
        videoRecorder.stop();   // onstop assembles the blob
    } else {
        resetVideoRecordingUI();
    }
}

if (btnRecordVideo) {
    btnRecordVideo.addEventListener('click', () => {
        if (isRecordingVideo) stopVideoRecording();
        else                  startVideoRecording();
    });
}

if (videoFile) {
    videoFile.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        videoBlob = file;
        if (videoFileName) videoFileName.textContent = file.name;
        setVideoStatus('File selected — ready to analyze');
        if (btnAnalyzeVideo) btnAnalyzeVideo.disabled = false;
    });
}

if (btnAnalyzeVideo) {
    btnAnalyzeVideo.addEventListener('click', async () => {
        if (!videoBlob) return;

        setVideoStatus('📤 Uploading and analysing — this takes a few seconds...');
        showLoading(true, 'Analysing video — face, voice and words...');
        btnAnalyzeVideo.disabled = true;
        if (btnRecordVideo) btnRecordVideo.disabled = true;

        const fd = new FormData();
        const name = (videoFileName && videoFileName.textContent !== 'No file chosen')
            ? videoFileName.textContent : 'recording.webm';
        fd.append('file', videoBlob, name);
        // Keeps the server's emotion memory alive across turns.
        if (currentSessionId) fd.append('session_id', currentSessionId);

        try {
            // No Content-Type header — the browser sets the multipart boundary.
            const res = await fetch(`${API_BASE}/analyze/video`, {
                method:  'POST',
                headers: { 'X-API-Key': SESSION_API_KEY },
                body:    fd,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            setVideoStatus('✅ Analysis complete');
            renderResults(await res.json());
        } catch (err) {
            setVideoStatus('❌ Analysis failed');
            showToast(err.message);
        } finally {
            showLoading(false);
            btnAnalyzeVideo.disabled = false;
            if (btnRecordVideo) btnRecordVideo.disabled = false;
        }
    });
}

// ============================================================================
// Option 4: Live Call (WebSocket)
// ============================================================================
// The browser captures mic + camera and STREAMS them to the server. Previously
// this tab opened getUserMedia purely to animate a level meter while the SERVER
// recorded from its own microphone - so it only ever worked for one person,
// sitting at the machine running the server.
//
// Wire protocol, matching routers/stream.py:
//   binary  0x01 + Int16LE PCM @ 16 kHz mono
//   binary  0x02 + JPEG bytes
//   text    JSON control both ways

const MSG_AUDIO = 0x01;
const MSG_VIDEO = 0x02;

// Frames per second sent to the server.
//
// The SERVER decides this and announces it in the handshake — it smooths face
// results against this rate, so if the two sides disagreed it would silently
// average over the wrong span of time. This is only the fallback used if the
// handshake somehow arrives without it.
let VIDEO_FPS = 3;

let ws               = null;
let liveStream       = null;   // the MediaStream from getUserMedia
let liveAudioCtx     = null;
let liveWorkletNode  = null;
let liveVideoTimer   = null;
let liveCanvas       = null;
let liveCtx2d        = null;
let liveVideoEl      = null;
let liveMicStopFn    = null;

function setStreamStatus(text) {
    if (streamStatusText) streamStatusText.textContent = text;
}

function stopStreamMicMonitor() {
    if (liveMicStopFn) { liveMicStopFn(); liveMicStopFn = null; }
    if (streamMicLevel)  streamMicLevel.style.width = '0%';
    if (micLevelWrapper) micLevelWrapper.style.display = 'none';
}

async function teardownLiveCapture() {
    if (liveVideoTimer) { clearInterval(liveVideoTimer); liveVideoTimer = null; }

    if (liveWorkletNode) {
        try { liveWorkletNode.port.onmessage = null; liveWorkletNode.disconnect(); } catch (_) {}
        liveWorkletNode = null;
    }
    if (liveAudioCtx) {
        try { await liveAudioCtx.close(); } catch (_) {}
        liveAudioCtx = null;
    }
    if (liveStream) {
        liveStream.getTracks().forEach(t => t.stop());
        liveStream = null;
    }
    if (liveVideoEl) {
        try { liveVideoEl.pause(); liveVideoEl.srcObject = null; } catch (_) {}
        liveVideoEl = null;
    }
    if (streamCameraPreview) {
        streamCameraPreview.srcObject     = null;
        streamCameraPreview.style.display = 'none';
    }
    if (streamCameraPlaceholder) streamCameraPlaceholder.style.display = 'flex';
    liveCanvas = null;
    liveCtx2d = null;
    stopStreamMicMonitor();
}

function disconnectWebSocket() {
    if (ws) {
        try {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop' }));
            }
            ws.close();
        } catch (_) {}
        ws = null;
    }
    teardownLiveCapture();
    showLoading(false);
    btnConnectStream.style.display      = 'inline-block';
    btnConnectStream.disabled           = false;
    btnDisconnectStream.style.display   = 'none';
    streamStatusContainer.style.display = 'none';
}

async function startLiveCapture() {
    // audio:true AND video:true - the video half is what the old code never
    // actually transmitted.
    liveStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount:     1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl:  true,
        },
        video: { width: 480, height: 360, facingMode: 'user' },
    });

    // -- Preview + mic meter --------------------------------------------------
    if (streamCameraPreview) {
        streamCameraPreview.srcObject     = liveStream;
        streamCameraPreview.style.display = 'block';
        if (streamCameraPlaceholder) streamCameraPlaceholder.style.display = 'none';
        streamCameraPreview.play().catch(() => {});
    }
    if (streamMicLevel && micLevelWrapper) {
        micLevelWrapper.style.display = 'flex';
        liveMicStopFn = createMicLevelMonitor(liveStream, (level) => {
            streamMicLevel.style.width = level + '%';
        });
    }

    // -- Audio: 16 kHz PCM via AudioWorklet -----------------------------------
    // The context is constructed AT 16 kHz so the browser resamples natively.
    // The server does not resample, and feeding the wrong rate to SpeechBrain
    // produces a confident wrong answer rather than an error.
    liveAudioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
    });
    await liveAudioCtx.audioWorklet.addModule('pcm-worklet.js');

    const source = liveAudioCtx.createMediaStreamSource(liveStream);
    liveWorkletNode = new AudioWorkletNode(liveAudioCtx, 'pcm-16k');

    liveWorkletNode.port.onmessage = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const pcm = new Uint8Array(event.data);
        const frame = new Uint8Array(pcm.length + 1);
        frame[0] = MSG_AUDIO;
        frame.set(pcm, 1);
        ws.send(frame);
    };

    source.connect(liveWorkletNode);
    // Deliberately NOT connected to destination - that would play the caller's
    // own microphone back through their speakers. A worklet keeps running
    // without a sink, unlike ScriptProcessorNode.

    // -- Video: JPEG stills on an interval ------------------------------------
    liveVideoEl = document.createElement('video');
    liveVideoEl.srcObject = liveStream;
    liveVideoEl.muted = true;
    liveVideoEl.playsInline = true;
    await liveVideoEl.play().catch(() => {});

    liveCanvas = document.createElement('canvas');
    liveCanvas.width  = 480;
    liveCanvas.height = 360;
    const ctx2d = liveCanvas.getContext('2d');

    liveCtx2d = ctx2d;
    // The timer is NOT started here. It starts once the server has told us the
    // rate it expects, in startVideoSending() below.
}

// Begins sending frames at the rate the SERVER asked for.
function startVideoSending(fps) {
    if (liveVideoTimer) clearInterval(liveVideoTimer);
    if (fps && fps > 0) VIDEO_FPS = fps;

    liveVideoTimer = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!liveVideoEl || liveVideoEl.readyState < 2) return;
        if (!liveCanvas || !liveCtx2d) return;

        liveCtx2d.drawImage(liveVideoEl, 0, 0, liveCanvas.width, liveCanvas.height);
        liveCanvas.toBlob(async (blob) => {
            if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
            const buf = new Uint8Array(await blob.arrayBuffer());
            const frame = new Uint8Array(buf.length + 1);
            frame[0] = MSG_VIDEO;
            frame.set(buf, 1);
            ws.send(frame);
        }, 'image/jpeg', 0.6);
    }, Math.round(1000 / VIDEO_FPS));
}

btnConnectStream.addEventListener('click', async () => {
    btnConnectStream.disabled = true;
    streamStatusContainer.style.display = 'flex';
    setStreamStatus('Requesting camera and microphone...');

    try {
        await startLiveCapture();
    } catch (err) {
        setStreamStatus('Could not start capture');
        showToast('Camera or microphone access was denied.');
        await teardownLiveCapture();
        btnConnectStream.disabled = false;
        streamStatusContainer.style.display = 'none';
        return;
    }

    setStreamStatus('Connecting...');

    const apiUrl  = new URL(API_BASE);
    const wsProto = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
    const params  = new URLSearchParams();
    if (currentSessionId) params.set('session_id', currentSessionId);
    // Browsers cannot set headers on a WebSocket handshake, so the key travels
    // as a query parameter. The server accepts either.
    if (SESSION_API_KEY) params.set('api_key', SESSION_API_KEY);

    ws = new WebSocket(wsProto + '//' + apiUrl.host + '/ws/stream?' + params.toString());
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        btnConnectStream.style.display    = 'none';
        btnDisconnectStream.style.display = 'inline-block';
        btnConnectStream.disabled         = false;
        setStreamStatus('Live - start speaking');
    };

    ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); }
        catch (e) { console.error('[Live] bad frame:', e); return; }

        if (data.type === 'error') {
            showToast(data.message || 'Streaming error.');
            disconnectWebSocket();
            return;
        }

        if (data.type === 'status') {
            if (data.code === 'CONNECTED') {
                // The server owns the frame rate; follow what it asked for.
                startVideoSending(data.config && data.config.target_fps);
            }

            // TURN_TOO_LONG used to be here. The server can never send it: the
            // turn detector closes and resets a turn at 20 seconds, so the
            // buffer check that raised it was unreachable and has been removed.
            // A long speaker simply gets their turn analysed at the 20s mark.
            const messages = {
                CONNECTED:   'Live - start speaking',
                ANALYZING:   'Analysing your turn...',
                BUSY:        'Skipped - still working on the previous turn',
                TURN_FAILED: 'That turn could not be analysed',
            };
            setStreamStatus(messages[data.code] || data.message || 'Connected');
            return;
        }

        // Anything carrying a session_id is a real result payload.
        if (data.session_id) {
            renderResults(data);
            setStreamStatus('Result in - keep talking');
        }
    };

    ws.onerror = () => {
        showToast(
            SESSION_API_KEY
                ? 'Live connection failed. Check that your API key is valid.'
                : 'Live connection failed.'
        );
    };

    ws.onclose = () => { disconnectWebSocket(); };
});

btnDisconnectStream.addEventListener('click', disconnectWebSocket);

// ============================================================================
// Init
// ============================================================================
initTheme();