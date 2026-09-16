"""
tests/test_video_upload.py
==========================
Guards POST /analyze/video against the kinds of input a real browser sends.

The file-type gate is the specific thing under test here. It was written against
tidy MIME strings like "video/webm" and rejected every actual in-browser
recording, because MediaRecorder sends "video/webm;codecs=vp9,opus" — the type
WITH parameters. That reached a user before any test did, so it gets a test.

Run:  python -m pytest tests/test_video_upload.py -v
"""

import os
import sys
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# The content types browsers actually send
# ══════════════════════════════════════════════════════════════════════════════

# Every one of these is a real MediaRecorder output string. The ones carrying
# ";codecs=..." are what Chrome and Firefox produce; bare "video/mp4" is Safari.
BROWSER_CONTENT_TYPES = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm; codecs=vp8,opus",   # some browsers include a space
    "video/webm",
    "video/mp4",
    "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
    "video/quicktime",               # .mov from an iPhone camera roll
    "application/octet-stream",      # some file pickers send nothing better
]

REJECTED_CONTENT_TYPES = [
    ("text/plain",       "notes.txt"),
    ("application/pdf",  "paper.pdf"),
    ("image/png",        "screenshot.png"),
    ("audio/wav",        "clip.wav"),   # belongs on /analyze/voice
]


def _normalise(content_type: str) -> str:
    """Mirror of the router's normalisation, kept in one place for the test."""
    return (content_type or "").split(";")[0].strip().lower()


@pytest.mark.parametrize("content_type", BROWSER_CONTENT_TYPES)
def test_browser_content_types_pass_the_gate(content_type):
    """
    A recording from any mainstream browser must get past the type check.

    This asserts on the gate directly rather than through a request, so it stays
    fast and needs no models — the point is the string handling.
    """
    from routers.video import _ALLOWED_TYPES

    assert _normalise(content_type) in _ALLOWED_TYPES, (
        f"'{content_type}' would be rejected before ffprobe ever sees it. "
        f"Normalised to '{_normalise(content_type)}', which is not in the "
        f"allow-list. Real browser recordings send parameters on the MIME type."
    )


@pytest.mark.parametrize("content_type,filename", REJECTED_CONTENT_TYPES)
def test_non_video_types_are_rejected(content_type, filename):
    """The gate must still refuse things that plainly are not video."""
    from routers.video import _ALLOWED_TYPES, _ALLOWED_SUFFIXES

    passes_type   = _normalise(content_type) in _ALLOWED_TYPES
    passes_suffix = filename.lower().endswith(_ALLOWED_SUFFIXES)

    assert not (passes_type or passes_suffix), \
        f"'{content_type}' / '{filename}' should not reach the decoder"


def test_recording_filenames_carry_an_extension():
    """
    The frontend must name a recording with a real extension.

    The gate accepts a known MIME type OR a known suffix. A blob named
    'browser_recording' with no extension fails the suffix half, so if the MIME
    check ever regresses there is nothing to fall back on — which is exactly how
    the original bug got through.
    """
    app_js = os.path.join(PROJECT_ROOT, "frontend", "app.js")
    with open(app_js, encoding="utf-8") as f:
        source = f.read()

    assert "browser_recording.${ext}" in source, (
        "The recorded-video filename lost its extension. It must include one, "
        "so the suffix check can accept a recording even if the MIME check does not."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Live round-trip, with a real generated video
# ══════════════════════════════════════════════════════════════════════════════

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_browser_recording_round_trip(tmp_path):
    """
    Posts a real WebM with the exact Content-Type Chrome sends.

    This is the end-to-end version of the bug: before the fix it returned 400
    with "Unsupported file type 'video/webm;codecs=vp9,opus'".
    """
    from fastapi.testclient import TestClient
    from src.core.config import API_KEYS
    import api


    clip = tmp_path / "clip.webm"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
        "-c:v", "libvpx", "-b:v", "300k", "-c:a", "libopus",
        str(clip),
    ], check=True, timeout=120)

    headers = {"X-API-Key": next(iter(API_KEYS))} if API_KEYS else {}

    with TestClient(api.app) as client:
        with open(clip, "rb") as f:
            response = client.post(
                "/analyze/video",
                files={"file": ("browser_recording.webm", f, "video/webm;codecs=vp9,opus")},
                headers=headers,
            )

    # 200 with a payload, or 422 when the synthetic clip has no face and no
    # speech. Either proves the file got past the type gate — which is the point.
    assert response.status_code in (200, 422), (
        f"Expected the upload to reach the decoder, got "
        f"{response.status_code}: {response.text[:300]}"
    )
    assert "Unsupported file type" not in response.text, (
        "The MIME-parameter regression is back — browser recordings are being "
        "rejected before decoding."
    )
