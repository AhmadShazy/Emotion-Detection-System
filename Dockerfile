# ─────────────────────────────────────────────────────────────────────────────
# Humanoid Assistant — input module
#
# Builds the TEXT-ONLY deployment: /analyze/text plus the /mock/* contract
# sandbox. That is what the LLM team needs to build against, and it needs no
# microphone, no camera and no OpenFace binary — so it runs on ordinary Linux
# hosting.
#
# Full mode (voice / multimodal / live) is NOT buildable this way yet: it
# depends on the server owning a microphone and a webcam, and on a Windows-only
# OpenFace executable. Removing those constraints is Phases 3 and 4.
#
# Build:  docker build -t humanoid-input .
# Run:    docker run -p 7860:7860 -e API_KEYS=your-key humanoid-input
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Hugging Face Spaces runs containers as UID 1000 and expects a writable home.
# Creating the user explicitly keeps this image portable across hosts that do
# the same, and avoids running as root anywhere else.
RUN useradd -m -u 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The project logs emoji and box-drawing characters. Containers default to
    # ASCII, which would raise UnicodeEncodeError on the first log line.
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    # matplotlib arrives as a transitive dependency and probes for a display.
    MPLBACKEND=Agg

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# requirements-text.txt is the verified minimal set — see the notes in that file
# before trimming it. The CPU torch index avoids ~2 GB of unused CUDA libraries.
COPY --chown=appuser:appuser requirements-text.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-text.txt

# ── Application ──────────────────────────────────────────────────────────────
COPY --chown=appuser:appuser api.py            ./
COPY --chown=appuser:appuser routers/          ./routers/
COPY --chown=appuser:appuser schemas/          ./schemas/
COPY --chown=appuser:appuser src/              ./src/
COPY --chown=appuser:appuser frontend/         ./frontend/
COPY --chown=appuser:appuser contract/         ./contract/

USER appuser

# ── Pre-download RoBERTa into the image ──────────────────────────────────────
# Without this the model is fetched on first boot, so the very first request
# waits ~30s and a cold restart repeats it. Baking it in trades image size for
# a fast, predictable start.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
m='SamLowe/roberta-base-go_emotions'; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m); \
print('RoBERTa cached into image')"

# ── Runtime configuration ────────────────────────────────────────────────────
# Text-only is the only mode this image can serve.
ENV TEXT_ONLY_MODE=true \
    # 7860 is the Hugging Face Spaces default. Other platforms inject $PORT.
    PORT=7860

EXPOSE 7860

# API_KEYS is deliberately NOT set here. Provide it at runtime:
#   docker run -e API_KEYS=...          or the platform's secrets UI.
# Leaving it unset starts the server with authentication disabled and prints a
# loud warning.

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; \
urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",7860)}/health').read()"

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
