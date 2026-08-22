"""
src/video/ingest.py
===================
Turns a user-supplied video file into the two things the analysis pipeline
needs: a 16 kHz mono WAV, and a directory of sampled RGB frames.

The server never opens a camera or a microphone. Everything here operates on a
file the browser sent, which is what makes the hosted multi-user model possible.

Why ffmpeg's CLI rather than a Python decoding library
------------------------------------------------------
openai-whisper already shells out to the ffmpeg binary (see whisper/audio.py),
so it is an accepted dependency and adds nothing new. More importantly it keeps
ONE decoder in the system: a file that transcribes is a file that yields frames.
Bundled-libav libraries (PyAV, OpenCV) each carry their own build and can
disagree about what is decodable.

`-vf fps=N` also normalises variable-frame-rate input to constant. Browser
MediaRecorder output is VFR, and MediaPipe's video mode rejects non-monotonic
timestamps, so this conversion is required rather than merely convenient.
"""

import os
import glob
import shutil
import subprocess

# Analysis window. Anything past this is ignored — it bounds both processing
# time and disk, and stops an unparseable duration from running away.
MAX_VIDEO_SECONDS = 30

# Frames sampled per second. The face reading is aggregated over the whole clip,
# so decoding every frame of 30 fps footage buys almost nothing for 10x the
# cost. 5 fps gives 150 frames across the full window.
FRAME_SAMPLE_FPS = 5

# Audio format is NOT negotiable. src/ser/ser_engine.py reads the WAV with
# soundfile and passes it straight to a wav2vec2 model that expects 16 kHz —
# there is no resampling anywhere in that path. Feed it 48 kHz and it returns a
# confident WRONG label rather than raising, which is the worst failure mode.
AUDIO_SAMPLE_RATE = 16000

_FFMPEG_TIMEOUT = 120


class VideoIngestError(Exception):
    """Raised when a file cannot be probed or decoded."""


def _run(cmd: list, timeout: int = _FFMPEG_TIMEOUT) -> subprocess.CompletedProcess:
    """
    Runs a command to completion.

    Always waits, never leaves a bare Popen. On Windows an open handle inside
    the job directory makes the cleanup rmtree fail silently, leaking the whole
    directory.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        # ffmpeg tries to read stdin for interactive prompts; without this it
        # can block forever in a server process.
        stdin=subprocess.DEVNULL,
    )


def probe_streams(path: str) -> dict:
    """
    Reports which streams a file actually contains.

    ffprobe doubles as the validity gate: random bytes and truncated uploads
    fail here cleanly rather than deeper in the pipeline. Never branch on the
    filename or the browser's Content-Type — both are hints, not evidence.

    Returns {"video": bool, "audio": bool}.
    """
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        path,
    ], timeout=30)

    if result.returncode != 0:
        raise VideoIngestError(
            "Could not read this file as video. It may be corrupt, empty, or "
            "not a video at all."
        )

    kinds = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return {"video": "video" in kinds, "audio": "audio" in kinds}


def extract(path: str, job_dir: str) -> dict:
    """
    Splits a video into a WAV and a directory of JPEG frames.

    Both outputs come from a single ffmpeg invocation so the file is decoded
    once rather than twice.

    Returns {"audio_path": str | None, "frame_paths": [str], "has_audio": bool}.
    audio_path is None when the source carries no audio track — a legitimate
    case (a silent clip still has a face to read), not an error.
    """
    streams = probe_streams(path)

    if not streams["video"]:
        raise VideoIngestError(
            "That file has no video track. Use the Voice tab for audio-only "
            "recordings."
        )

    frames_dir = os.path.join(job_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    audio_path = os.path.join(job_dir, "audio.wav")

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        # Trim BEFORE decoding the rest, so a long upload costs no extra work.
        "-t", str(MAX_VIDEO_SECONDS),
        "-i", path,
    ]

    if streams["audio"]:
        # Only add the audio mapping when a track exists. `-map 0:a:0` against a
        # silent video does not merely skip the audio — it fails the ENTIRE
        # command ("Stream map matches no streams"), so no frames are produced
        # either.
        cmd += [
            "-map", "0:a:0", "-vn",
            "-c:a", "pcm_s16le",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", "1",
            audio_path,
        ]

    cmd += [
        "-map", "0:v:0", "-an",
        "-vf", f"fps={FRAME_SAMPLE_FPS}",
        "-q:v", "3",
        os.path.join(frames_dir, "frame_%05d.jpg"),
    ]

    result = _run(cmd)

    if result.returncode != 0:
        raise VideoIngestError(
            f"Could not decode this video. ffmpeg said: "
            f"{result.stderr.strip()[:300]}"
        )

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))

    if not frame_paths:
        raise VideoIngestError(
            "No frames could be read from that video, even though it reported a "
            "video track."
        )

    has_audio = streams["audio"] and os.path.isfile(audio_path)

    return {
        "audio_path":  audio_path if has_audio else None,
        "frame_paths": frame_paths,
        "has_audio":   has_audio,
    }


def cleanup(job_dir: str) -> None:
    """
    Removes a job directory and everything under it.

    ignore_errors survives a partially-built tree, which happens when a request
    is rejected mid-write.
    """
    shutil.rmtree(job_dir, ignore_errors=True)
