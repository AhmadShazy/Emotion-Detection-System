"""
routers/video.py
================
POST /analyze/video

Accepts a video the user recorded in the browser or picked from their device,
and returns the standard emotion payload.

This replaces the old start/stop multimodal mode, which opened the SERVER's own
webcam and microphone. That design could only ever serve one person sitting at
the machine running it. Here the client supplies the media and the server owns
no capture device at all, which is what makes a hosted multi-user service
possible.

The analysis itself is unchanged: the audio track goes through the existing
voice pipeline, the frames go through the face analyser, and both feed the same
fusion engine.
"""

import os
import sys
import shutil
import tempfile
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from schemas.emotion import UnifiedEmotionResponse
from src.interactive_modes import process_voice_pipeline, _is_hallucination
from src.streaming.unified_pipeline import process_and_print_unified_json
from src.video.ingest import extract, cleanup, VideoIngestError, MAX_VIDEO_SECONDS

router = APIRouter()

# Where per-request working directories live. Each request gets its own,
# created with mkdtemp so two concurrent uploads can never collide.
JOBS_ROOT = os.path.join(PROJECT_ROOT, "data", "jobs")

# 64 MB. Roughly 30 seconds of 720p from a phone, with headroom.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# Read the upload in chunks so an oversized file is rejected without ever being
# held in memory.
_CHUNK = 1024 * 1024

_ALLOWED_TYPES = {
    "video/webm", "video/mp4", "video/quicktime", "video/x-matroska",
    "video/ogg", "application/octet-stream",
}
_ALLOWED_SUFFIXES = (".webm", ".mp4", ".mov", ".mkv", ".ogg", ".m4v")


async def _save_upload(file: UploadFile, dest: str) -> int:
    """
    Streams the upload to disk, aborting if it exceeds the size cap.

    Returns the number of bytes written. Raises 413 rather than letting a large
    file consume memory or disk.
    """
    written = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Video is larger than "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                        f"Record a shorter clip, or use a lower resolution."
                    ),
                )
            out.write(chunk)
    return written


@router.post(
    "/video",
    response_model=UnifiedEmotionResponse,
    summary="Video Emotion Analysis",
    description=(
        "Upload a video recorded in the browser or chosen from the device. "
        "The server extracts the audio track and the video frames, analyses "
        "voice, speech and facial expression, and returns the fused emotion "
        f"payload. Only the first {MAX_VIDEO_SECONDS} seconds are analysed."
    ),
)
async def analyze_video(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    content_type = (file.content_type or "").lower()
    filename     = (file.filename or "").lower()

    # A hint, not evidence — ffprobe is the authority on whether this decodes.
    if content_type not in _ALLOWED_TYPES and not filename.endswith(_ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Upload a video (.webm, .mp4, .mov)."
            ),
        )

    os.makedirs(JOBS_ROOT, exist_ok=True)

    # Created BEFORE the try, so the finally can never reference an unbound
    # name — the same shape routers/voice.py uses for its temp WAV.
    job_dir = tempfile.mkdtemp(prefix="video_", dir=JOBS_ROOT)
    upload_path = os.path.join(job_dir, "upload.bin")

    try:
        size = await _save_upload(file, upload_path)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # ── Split into audio + frames (blocking, so off the event loop) ──────
        try:
            media = await asyncio.to_thread(extract, upload_path, job_dir)
        except VideoIngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def _analyse():
            # ── Face ────────────────────────────────────────────────────────
            from src.faceexpression.mediapipe_analyzer import analyze_frames
            _timeline, face_state = analyze_frames(media["frame_paths"])

            # ── Voice + speech + text, via the existing pipeline ────────────
            text_state = voice_state = None
            stt_result = ser_result = "N/A"

            if media["has_audio"]:
                text_state, voice_state, stt_result, ser_result = (
                    process_voice_pipeline(media["audio_path"])
                )

            effective_text = (
                stt_result
                if stt_result and stt_result != "N/A"
                   and not _is_hallucination(stt_result)
                else ""
            )

            # Nothing usable at all — no face anywhere, and no speech.
            if face_state is None and not effective_text and voice_state is None:
                return None

            return process_and_print_unified_json(
                text_state=text_state,
                voice_state=voice_state,
                face_state=face_state,
                raw_text=effective_text,
                voice_emo_raw=ser_result if ser_result != "N/A" else "neutral",
                face_emo_raw=face_state["emotion"] if face_state else "neutral",
                session_id=session_id,
            )

        payload = await asyncio.to_thread(_analyse)

        if payload is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Nothing could be analysed in that video — no face was "
                    "visible in any frame and no speech was detected. Check "
                    "lighting, that your face is in shot, and that the "
                    "recording captured audio."
                ),
            )

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis pipeline error: {exc}",
        ) from exc
    finally:
        # One directory holds every artefact, so one removal cleans up
        # completely — including when the request failed part-way.
        cleanup(job_dir)


def sweep_stale_jobs(max_age_seconds: int = 3600) -> int:
    """
    Removes job directories left behind by a process that was killed mid-request.

    The finally block above handles every normal path, but cannot survive a
    SIGKILL, an OOM kill, or a host reboot. Called from the API's startup.
    Returns how many directories were removed.
    """
    import time

    if not os.path.isdir(JOBS_ROOT):
        return 0

    now = time.time()
    removed = 0

    for name in os.listdir(JOBS_ROOT):
        path = os.path.join(JOBS_ROOT, name)
        try:
            if os.path.isdir(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            pass

    return removed
