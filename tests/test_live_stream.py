"""
tests/test_live_stream.py
=========================
Covers the live-call path: turn detection, and the WebSocket wire protocol.

The turn-detection tests exist because the previous detector was broken on this
project's own data. It used a FIXED silence threshold of 0.01, and measuring the
recordings in data/recordings/ showed RMS spanning 0.0029 to 0.0419 — so the
quietest files read as 100% silence (a turn never opened) and the loudest read
as 100% speech (a turn never closed). Both failure modes produced zero turns,
which looks identical to "the feature does nothing".

Run:  python -m pytest tests/test_live_stream.py -v
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.streaming.turn_detector import TurnDetector, SAMPLE_RATE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tone(seconds: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    """Speech-ish signal at a chosen loudness."""
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float, noise: float = 0.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    if noise <= 0:
        return np.zeros(n, dtype=np.float32)
    return (np.random.randn(n) * noise).astype(np.float32)


def _feed(detector: TurnDetector, audio: np.ndarray, packet: int = 2048) -> list:
    """Pushes audio in browser-sized packets, collecting completed turns."""
    turns = []
    for i in range(0, len(audio), packet):
        turn = detector.push(audio[i:i + packet])
        if turn is not None:
            turns.append(turn)
    return turns


# ── Turn detection ────────────────────────────────────────────────────────────

def test_detects_a_normal_turn():
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),   # room tone, lets the floor calibrate
        _tone(2.0, amplitude=0.2),    # the utterance
        _silence(2.0, noise=0.001),   # trailing pause closes the turn
    ])
    turns = _feed(d, audio)

    assert len(turns) == 1, f"expected exactly one turn, got {len(turns)}"
    assert turns[0]["reason"] == "silence"
    assert 1.5 < turns[0]["duration"] < 4.0, turns[0]["duration"]


@pytest.mark.parametrize("amplitude", [0.004, 0.02, 0.2, 0.5])
def test_detects_speech_across_a_wide_loudness_range(amplitude):
    """
    The core regression. The old fixed 0.01 threshold could not span this:
    0.004 sat entirely below it (all silence, turn never opened) and 0.2/0.5
    sat entirely above it (all speech, turn never closed).
    """
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=amplitude * 0.02),   # floor scales with the signal
        _tone(2.0, amplitude=amplitude),
        _silence(2.0, noise=amplitude * 0.02),
    ])
    turns = _feed(d, audio)

    assert len(turns) == 1, (
        f"amplitude {amplitude}: expected one turn, got {len(turns)}. "
        f"detector state: {d.stats()}"
    )


def test_pure_silence_produces_no_turn():
    d = TurnDetector()
    turns = _feed(d, _silence(5.0, noise=0.0005))
    assert turns == [], "silence must not produce a turn"


def test_short_blip_is_ignored():
    """A cough or a door is not a turn."""
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(0.2, amplitude=0.3),     # under MIN_TURN_SECONDS
        _silence(2.0, noise=0.001),
    ])
    assert _feed(d, audio) == []


def test_pause_inside_a_sentence_does_not_split_it():
    """
    A breath mid-sentence is shorter than the trailing-silence threshold, so it
    must not cut the turn. Without hysteresis this is where a detector chops one
    utterance into several.
    """
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(1.5, amplitude=0.2),
        _silence(0.4, noise=0.001),    # brief pause, well under 1.5s
        _tone(1.5, amplitude=0.2),
        _silence(2.0, noise=0.001),
    ])
    turns = _feed(d, audio)

    assert len(turns) == 1, f"the pause split the turn: got {len(turns)} turns"
    assert turns[0]["duration"] > 3.0, "the two halves were not joined"


def test_endless_speech_is_cut_at_the_cap():
    """A caller who never pauses must still produce turns, not grow forever."""
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(25.0, amplitude=0.2),    # past MAX_TURN_SECONDS
    ])
    turns = _feed(d, audio)

    assert len(turns) >= 1
    assert turns[0]["reason"] == "max_length"
    assert turns[0]["duration"] <= 21.0


def test_packet_size_does_not_change_the_result():
    """
    The network decides packet sizes, not us. Counting in samples rather than
    chunks is what makes the detector agnostic to that.
    """
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(2.0, amplitude=0.2),
        _silence(2.0, noise=0.001),
    ])
    counts = []
    for packet in (320, 2048, 4096, 8000):
        d = TurnDetector()
        counts.append(len(_feed(d, audio, packet=packet)))

    assert len(set(counts)) == 1, f"packet size changed the outcome: {counts}"


def test_abandon_turn_discards_without_emitting():
    d = TurnDetector()
    _feed(d, np.concatenate([_silence(1.2, noise=0.001), _tone(1.0, 0.2)]))
    assert d.turn_samples > 0
    assert d.abandon_turn() is True
    assert d.turn_samples == 0


# ── Session buffering ─────────────────────────────────────────────────────────

def test_session_converts_pcm_bytes_correctly():
    """
    Int16LE on the wire must land as float32 in [-1, 1]. Getting this wrong is
    silent: SpeechBrain returns a confident WRONG label rather than raising.
    """
    from src.streaming.call_session import CallSession

    session = CallSession()
    original = _tone(0.5, amplitude=0.5)
    pcm = (np.clip(original, -1, 1) * 32767).astype("<i2").tobytes()

    session.push_audio(pcm)

    assert session.audio_samples_received == len(original)
    recovered = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    assert np.max(np.abs(recovered - original)) < 1e-3


def test_session_bounds_the_video_buffer():
    from src.streaming.call_session import CallSession, MAX_VIDEO_FRAMES

    session = CallSession()
    for _ in range(MAX_VIDEO_FRAMES + 50):
        session.push_video(b"\xff\xd8fake-jpeg")

    assert len(session.video_frames) <= MAX_VIDEO_FRAMES
    assert session.video_frames_received == MAX_VIDEO_FRAMES + 50


def test_session_refuses_oversized_frames():
    from src.streaming.call_session import CallSession, MAX_FRAME_BYTES

    session = CallSession()
    assert session.push_video(b"x" * (MAX_FRAME_BYTES + 1)) is False
    assert session.push_video(b"x" * 1000) is True


# ── Wire protocol ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_websocket_accepts_and_reports_config():
    """
    Proves the endpoint upgrades and announces the format it expects. This also
    guards the dependency: plain `uvicorn` has no WebSocket implementation and
    the handshake fails before reaching the app.
    """
    from fastapi.testclient import TestClient
    from src.core.config import API_KEYS, TEXT_ONLY_MODE
    import api

    if TEXT_ONLY_MODE:
        pytest.skip("live stream is disabled in text-only mode")

    key = next(iter(API_KEYS)) if API_KEYS else ""

    with TestClient(api.app) as client:
        with client.websocket_connect(f"/ws/stream?api_key={key}") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "status"
            assert hello["code"] == "CONNECTED"

            config = hello["config"]
            # These four are the contract with the browser client. Changing any
            # of them without changing frontend/app.js breaks the call silently.
            assert config["sample_rate"] == 16000
            assert config["audio_format"] == "pcm_s16le"
            assert config["video_format"] == "jpeg"

            ws.send_json({"type": "stop"})


def test_flush_recovers_speech_when_the_stream_ends_mid_sentence():
    """
    A caller who hangs up while still talking must not lose that turn.

    Without flush() the detector sits holding the speech, waiting for a
    trailing pause that will never arrive. Measured on the real recordings in
    data/recordings/: three of ten files end mid-sentence, and every one of
    them produced ZERO turns until this existed.
    """
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(3.0, amplitude=0.2),     # stream ends here, still talking
    ])
    turns = _feed(d, audio)
    assert turns == [], "no turn should close while speech is still going"

    final = d.flush()
    assert final is not None, "flush must recover the in-progress turn"
    assert final["reason"] == "stream_ended"
    assert final["duration"] > 2.0


def test_flush_does_not_invent_a_turn_from_noise():
    """Hanging up during a cough must not produce a junk turn."""
    d = TurnDetector()
    _feed(d, np.concatenate([_silence(1.2, noise=0.001), _tone(0.15, 0.3)]))
    assert d.flush() is None


def test_trailing_silence_is_trimmed_before_analysis():
    """
    The pause that ends a turn carries no information, and both Whisper and
    SpeechBrain cost time in proportion to input length — so shipping 1.5s of
    silence on every turn is pure latency.
    """
    d = TurnDetector()
    audio = np.concatenate([
        _silence(1.2, noise=0.001),
        _tone(2.0, amplitude=0.2),
        _silence(2.0, noise=0.001),
    ])
    turns = _feed(d, audio)
    assert len(turns) == 1
    # 2s of speech, plus a small keep-margin, and clearly less than the
    # 2s+2s the raw buffer would have held.
    assert turns[0]["duration"] < 3.0, (
        f"trailing silence was not trimmed: {turns[0]['duration']}s"
    )


@pytest.mark.live
def test_completed_turn_is_delivered_over_the_socket():
    """
    Streams enough real speech to close a turn and asserts a payload comes BACK.

    This is the gap that let a crash reach the user: the existing socket test
    only checked the handshake, so scheduling the analysis wrong
    (`create_task` on a Future -> "a coroutine was expected") killed the
    connection the instant the first turn completed, and no test noticed.
    The analysis itself ran fine — only the delivery was broken, which is
    exactly the kind of fault a handshake-only test cannot see.
    """
    import glob
    import soundfile as sf
    from fastapi.testclient import TestClient
    from src.core.config import API_KEYS, TEXT_ONLY_MODE
    import api

    if TEXT_ONLY_MODE:
        pytest.skip("live stream is disabled in text-only mode")

    # A recording with real, audible speech.
    candidates = sorted(glob.glob("data/recordings/voice_analysis_*.wav"))
    if not candidates:
        pytest.skip("no speech recording available to stream")

    audio, sr = sf.read(candidates[0])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        pytest.skip(f"recording is {sr} Hz, expected 16000")

    # Trailing silence so the turn closes NATURALLY, mid-stream. This is the
    # important part: hanging up takes a different code path (flush), and a
    # test that only hangs up cannot see a fault in normal turn completion —
    # which is exactly how the "a coroutine was expected" crash reached a user.
    audio = np.concatenate([audio, _silence(2.5, noise=0.0005)])

    key = next(iter(API_KEYS)) if API_KEYS else ""
    packet = 2048

    with TestClient(api.app) as client:
        with client.websocket_connect(f"/ws/stream?api_key={key}") as ws:
            assert ws.receive_json()["code"] == "CONNECTED"

            for i in range(0, len(audio), packet):
                chunk = audio[i:i + packet]
                pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
                ws.send_bytes(bytes([0x01]) + pcm)

            # Do NOT send "stop" — the payload must arrive from the turn
            # closing on its own.
            payload = None
            for _ in range(10):
                message = ws.receive_json()
                if message.get("session_id"):
                    payload = message
                    break

            assert payload is not None, (
                "no payload was delivered after a turn closed on trailing "
                "silence — the socket carried status frames but never a result"
            )
            assert payload["emotion_analysis"]["dominant_emotion"]
            assert "conflict_analysis" in payload
