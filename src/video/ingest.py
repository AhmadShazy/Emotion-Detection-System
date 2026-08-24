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

    Returns {"video": bool, "audio": bool, "kinds": [...], "detail": str}.
    """
    # Generous probe limits. The defaults (5 MB / 5 s) can miss the video stream
    # in a long recording from a phone, in a fragmented MP4, or in a MOV whose
    # index sits at the end of the file — ffprobe then exits 0 having found
    # nothing, which looks identical to "this has no video".
    result = _run([
        "ffprobe", "-v", "error",
        "-probesize", "100M",
        "-analyzeduration", "100M",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "csv=p=0",
        path,
    ], timeout=60)

    if result.returncode != 0:
        raise VideoIngestError(
            f"Could not read this file as video. It may be corrupt, empty, or "
            f"not a video at all. ffprobe: {result.stderr.strip()[:200]}"
        )

    kinds, described = set(), []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Fields come back as "codec_name,codec_type" — order follows the file,
        # so identify the type by matching rather than by position.
        parts = [p.strip() for p in line.split(",") if p.strip()]
        for kind in ("video", "audio", "subtitle", "data", "attachment"):
            if kind in parts:
                kinds.add(kind)
                other = [p for p in parts if p != kind]
                described.append(f"{kind} ({other[0]})" if other else kind)
                break

    return {
        "video":  "video" in kinds,
        "audio":  "audio" in kinds,
        "kinds":  sorted(kinds),
        "detail": ", ".join(described) if described else "no streams at all",
    }


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
        # Say what WAS found. "No video track" on a file the user knows is a
        # video is baffling on its own, and the detail is what makes it
        # diagnosable.
        raise VideoIngestError(
            f"No video track could be read from that file. "
            f"What was found: {streams['detail']}. "
            f"If this really is a video, the file may be partially uploaded or "
            f"in a container this server cannot read."
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


def ensure_16k_mono(wav_path: str) -> str:
    """
    Guarantees a WAV is 16 kHz mono, converting in place if it is not.

    Returns the path to use (the original when it was already correct, or a
    converted file alongside it).

    Why this has to exist: SpeechBrain's wav2vec2 expects 16 kHz and NOTHING in
    this project resamples. Feeding it 44.1 kHz does not raise — it returns a
    confident WRONG label. The video path gets 16 kHz from ffmpeg's -ar and the
    live path from the browser's AudioContext, but /analyze/voice accepts any
    .wav a user can produce, so it needs this guard.
    """
    import soundfile as sf

    try:
        info = sf.info(wav_path)
    except Exception as exc:
        raise VideoIngestError(f"Could not read that audio file: {exc}")

    if info.samplerate == AUDIO_SAMPLE_RATE and info.channels == 1:
        return wav_path

    converted = os.path.splitext(wav_path)[0] + "_16k.wav"
    result = _run([
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", wav_path,
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        converted,
    ], timeout=60)

    if result.returncode != 0 or not os.path.isfile(converted):
        raise VideoIngestError(
            f"Could not convert that audio to 16 kHz mono. "
            f"ffmpeg said: {result.stderr.strip()[:200]}"
        )

    print(f"[Audio] Converted {info.samplerate} Hz / {info.channels}ch -> 16 kHz mono")
    return converted
