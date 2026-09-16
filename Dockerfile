# ─────────────────────────────────────────────────────────────────────────────
# Humanoid Assistant — input module
#
# Builds FULL mode: text, voice, video upload and the live video call, plus the
# /mock/* contract sandbox. The server owns no microphone and no camera —
# everything is captured in the browser and uploaded — so nothing here needs a
# capture device, and any number of people can use one instance at once.
#
# Build:  docker build -t humanoid-input .
# Run:    docker run -p 8000:8000 -e PORT=8000 -e API_KEYS=your-key humanoid-input
#
# Verify it is self-contained (this is the point of baking the models in):
#         docker run --network=none -p 8000:8000 -e PORT=8000 \
#                    -e API_KEYS=k humanoid-input
#         Every model must load with no network at all.
#
# Expect roughly a 5 GB image and a 1.6 GB resident set. That trades disk for a
# predictable cold start: Cloud Run gives a container ~240 s to pass its startup
# probe, and downloading ~1.7 GB of models inside that window is not something
# to gamble a demo on. It also removes a real failure seen on a laptop, where
# Hugging Face answered model requests with HTTP 429 during startup.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System packages ──────────────────────────────────────────────────────────
# ffmpeg is NOT optional in full mode. src/video/ingest.py shells out to ffprobe
# and ffmpeg to validate uploads, extract frames and normalise audio to 16 kHz,
# and openai-whisper shells out to it again to decode audio inside
# model.transcribe(). Without it every voice and video request fails.
#
# libgl1 and libglib2.0-0 are for OpenCV, which mediapipe depends on. The
# non-headless opencv wheel links against libGL, so `import cv2` — and therefore
# all face analysis — fails on a bare slim image without them.
#
# libportaudio2 is here because mediapipe declares sounddevice~=0.5 as a hard
# dependency, so pip installs it whether or not this project wants it —
# excluding it from requirements.lock does not stop that. sounddevice loads
# libportaudio at import, and without the system library that import raises
# OSError. The server opens no audio device and never calls it deliberately,
# but anything that merely imports it would fail, so the ~100 KB is worth more
# than the argument.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000 and expects a writable home.
# Creating the user explicitly keeps this image portable across hosts that do
# the same, and avoids running as root anywhere else.
RUN useradd -m -u 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The project logs emoji and box-drawing characters. Containers default to
    # ASCII, which would raise UnicodeEncodeError on the first log line.
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    HOME=/home/appuser \
    # Every model cache lives under one tree inside the image. This matters
    # beyond tidiness: SpeechBrain's savedir is populated with SYMLINKS into the
    # Hugging Face cache, so the two must sit in the same image (and, on a build
    # host, the same layer) or the links dangle and SER fails to load.
    HF_HOME=/app/external/hf \
    # matplotlib arrives as a transitive dependency and probes for a display.
    MPLBACKEND=Agg

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# requirements.lock, not requirements.txt. The unpinned file resolves to
# different libraries on different days, and this system fails SILENTLY when
# that happens — model_registry swallows loader exceptions and /health reports
# "ok" regardless, so a bad resolve produces a server that starts, looks
# healthy, and answers "neutral" forever. See the header of requirements.lock.
COPY --chown=appuser:appuser requirements.lock ./

# These are three separate RUN steps on purpose, each with a BuildKit pip cache
# mount. Both details are about surviving a bad connection rather than style.
#
# Separate steps mean a failure part-way through does not throw away the work
# that already succeeded: torch alone is a 190 MB download, and as one combined
# step it was re-fetched from scratch on every retry.
#
# The cache mount keeps downloaded wheels OUTSIDE the image, in BuildKit's own
# cache, so a retry reuses them without re-downloading and the image carries no
# extra weight. PIP_NO_CACHE_DIR is deliberately not set — it would disable the
# very cache this relies on. On a link that has dropped to 19 kB/s mid-download,
# this is the difference between a retry costing seconds and costing half an
# hour.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# torch and torchaudio come from the CPU index FIRST. The PyPI Linux wheels
# bundle CUDA and add several GB to an image with no GPU to use it. Under
# PEP 440 the resulting 2.11.0+cpu satisfies the ==2.11.0 pin, so the pass over
# requirements.lock below leaves them alone rather than pulling the CUDA build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchaudio==2.11.0

# --retries and --timeout are not decoration. On a flaky link this step failed
# on ctranslate2 with "from versions: none" -- not a missing wheel (the cp310
# manylinux build exists and matches this image) but an index request that came
# back empty, which pip reports as though the package does not exist. Every
# other package in the same run downloaded normally. Ten retries and a longer
# timeout cost nothing on a good connection and stop a bad one from discarding
# half an hour of work over one dropped request.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 -r requirements.lock

# Fail the build rather than ship a CUDA torch by accident. The check is cheap
# and the mistake is expensive: several GB of libraries no CPU host can use.
RUN python -c "import torch; assert '+cpu' in torch.__version__, \
        f'expected a CPU build, got {torch.__version__}'; \
        print('torch', torch.__version__)"

# ── Models baked into the image ──────────────────────────────────────────────
# Only the handful of files the downloader itself imports are copied here, so
# this layer — the expensive one, ~1.7 GB — is not invalidated every time
# application code changes.
# WORKDIR created /app while USER was still root, so it is root-owned. The
# application runs as appuser and creates data/recordings, data/processed and
# data/jobs at startup (api.py, in the lifespan handler, before the models
# load) — unguarded, so a PermissionError there means uvicorn logs "Application
# startup failed" and the container never serves a request. Creating those
# directories now and chowning the whole of /app, not just external/, fixes the
# startup path and every later write into the workdir at once.
RUN mkdir -p /app/external \
             /app/data/recordings /app/data/processed /app/data/jobs \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/__init__.py            ./src/
COPY --chown=appuser:appuser src/core/__init__.py       ./src/core/
COPY --chown=appuser:appuser src/core/console.py        ./src/core/
# download_models.py imports this INSIDE download_speechbrain(), which is why it
# is missing from a glance at the top-of-file imports. Without it the model step
# dies with ModuleNotFoundError — after pip, and after three of the five models
# have already downloaded.
COPY --chown=appuser:appuser src/core/speechbrain_compat.py ./src/core/
COPY --chown=appuser:appuser scripts/download_models.py ./scripts/

USER appuser

# Exits non-zero if any of the five models fails, so a half-populated image can
# never be published. That is the whole point: a container that starts without
# its models does not crash, it serves "neutral" behind a green health check.
RUN python scripts/download_models.py

# ── Application ──────────────────────────────────────────────────────────────
# Last, because it changes most often and everything above should stay cached.
COPY --chown=appuser:appuser api.py    ./
COPY --chown=appuser:appuser routers/  ./routers/
COPY --chown=appuser:appuser schemas/  ./schemas/
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser contract/ ./contract/

# ── Runtime configuration ────────────────────────────────────────────────────
ENV PORT=8000 \
    # Thread counts are pinned because PyTorch and OpenMP size their pools from
    # the CPUs VISIBLE IN THE CONTAINER, and a 1-2 vCPU allocation on a shared
    # host commonly still reports the host's full core count. Unpinned, three
    # concurrent analysis stages each spawn 8-32 intra-op threads onto a
    # fraction of one core — the difference between "slower than the laptop"
    # and "unusable" on a system whose value proposition is live-call latency.
    #
    # These match the default INFERENCE_WORKERS=2. Raise them together with the
    # instance size, and measure rather than assuming: more concurrent model
    # instances on few cores has already been observed to be SLOWER, not faster.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    # The models are in the image, but from_pretrained() does not trust a
    # populated cache on its own: without this it still issues a HEAD request
    # per file to huggingface.co on every cold start. Under the CI check that
    # runs with --network=none those fail instantly and fall back to the cache,
    # so verification would exercise a path production never takes. In the
    # cases that matter — an egress blackhole where the HEAD hangs against
    # Cloud Run's 240 s startup probe, or an intercepting proxy whose SSLError
    # is re-raised rather than cached around — it lands in the loader's
    # swallow-everything except and serves "neutral" behind a green /health.
    #
    # It belongs HERE and not in the earlier ENV block: set before the download
    # step, it would forbid the very fetches that populate the cache.
    HF_HUB_OFFLINE=1

EXPOSE 8000

# API_KEYS is deliberately NOT set here. Provide it at runtime:
#   docker run -e API_KEYS=...          or the platform's secrets UI.
# Leaving it unset starts the server with authentication DISABLED and prints a
# loud banner saying so. There is no separate rule for local callers.

# Cloud Run and similar platforms ignore this and use their own probe, so point
# theirs at /health too. start-period covers model load from a warm image
# cache; the models are already on disk, so this is loading, not downloading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; \
urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health').read()"

# --workers 1 is deliberate and load-bearing, not a default left unexamined.
# Session state (src/streaming/unified_pipeline.py) and the live-call capacity
# counter (routers/stream.py) are both per-process, so a second worker would
# silently double MAX_ACTIVE_CALLS while doubling 1.6 GB of resident models.
# Scale with instances, and keep it to one — see BACKLOG.md.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
