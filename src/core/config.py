"""
src/core/config.py
==================
Central configuration loader.

Every setting the application reads comes from here. No other module should
touch os.environ directly, so there is exactly one place to look when behaviour
depends on the environment.

Values come from a .env file in the project root, or from real environment
variables when deployed (which is how hosting platforms inject secrets).
See .env.example for the full list.
"""

import os
from dotenv import load_dotenv

# Real environment variables win over .env, which is what you want in
# production: the platform's injected secrets should not be overridden by a
# stray file left in the image.
load_dotenv(override=False)


def _env(name: str, default: str = "") -> str:
    """
    Reads an environment variable, case-insensitively.

    Why case-insensitively: Windows environment variables are case-insensitive,
    so a .env typo like `API_KEYs=...` still resolves during local development.
    On Linux it does not — os.getenv("API_KEYS") simply returns nothing. For the
    auth settings that meant deploying to a Linux host could silently produce an
    empty key set and disable authentication, with no error anywhere. Matching
    case-insensitively makes both platforms behave identically.
    """
    value = os.getenv(name)
    if value is not None:
        return value

    target = name.lower()
    for key, val in os.environ.items():
        if key.lower() == target:
            return val

    return default


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)).strip())
    except ValueError:
        print(f"[Config] {name} is not a whole number — using {default}.")
        return default


# ── Mode ──────────────────────────────────────────────────────────────────────
# true  → only /analyze/text and /mock/* are served, and only RoBERTa is loaded.
#         Suitable for a small hosted instance the LLM team can build against.
# false → every mode, all four models. Media comes from the browser, so the
#         server needs no microphone, no camera and no OpenFace binary.
TEXT_ONLY_MODE: bool = _env_bool("TEXT_ONLY_MODE", False)

# ── Auth ──────────────────────────────────────────────────────────────────────
# A comma-separated list of accepted keys. These are values YOU choose — they
# are not issued by any third party. Give each teammate their own so one can be
# revoked without disturbing the others.
#
#   API_KEYS=<key-for-you>,<key-for-teammate>
#
# Generate a strong one with:
#   python -c "import secrets; print('hma_' + secrets.token_urlsafe(32))"
#
# Never commit real key values — not in .env (gitignored), and not in a comment
# here. An empty set disables authentication entirely; api.py prints a loud
# warning at startup in that case so it can never happen silently.
_raw_keys = _env("API_KEYS") or _env("API_KEY")
API_KEYS: set = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# ── LLM Forwarding ────────────────────────────────────────────────────────────
# Where completed emotion payloads are POSTed. Fire-and-forget: a slow or broken
# endpoint never blocks or fails the user's request.
LLM_ENDPOINT_URL: str = _env("LLM_ENDPOINT_URL")

# NOTE: there is deliberately no ALLOW_LOCALHOST setting. Authentication used to
# be skipped for callers on a loopback address, which meant the auth path was
# the one path local testing never exercised — and behind a cloud proxy the
# socket peer is the proxy, sometimes itself on loopback, so switching it on in
# a deployment would have opened everything rather than nothing. Running here
# and running on a server now take the same path: present a key.

# ── Capacity ──────────────────────────────────────────────────────────────────
# These were previously read straight from os.environ inside routers/stream.py
# and mediapipe_analyzer.py, which quietly broke the rule this module documents:
# there should be exactly one place to look when behaviour depends on the
# environment. They live here now.
#
# The defaults suit a 4-core CPU box. All three are hardware-shaped — raising
# them without more cores makes things SLOWER, because Whisper and SpeechBrain
# already saturate a core each and simply contend when run in parallel.

# How many live calls may be in progress at once. A caller beyond this is
# refused with AT_CAPACITY rather than accepted into unusable latency.
MAX_ACTIVE_CALLS: int = _env_int("MAX_ACTIVE_CALLS", 2)

# Worker threads available for turn analysis across ALL live calls.
INFERENCE_WORKERS: int = _env_int("INFERENCE_WORKERS", 2)

# MediaPipe face landmarker instances in the shared pool. Each is cheap
# (~90ms to build, 3.8MB) but they compete for the same cores as everything else.
FACE_POOL_WORKERS: int = _env_int("FACE_POOL_WORKERS", 2)

# Frames per second the browser sends during a live call.
#
# The SERVER owns this number and announces it in the WebSocket handshake, so
# the browser follows rather than deciding for itself. It was previously
# hardcoded in frontend/app.js, which meant the server was smoothing face
# results against an assumed rate it had no way to verify — change one side and
# the other silently smoothed over the wrong duration.
#
# The face reading is a majority vote over the whole turn, so a higher rate adds
# cost without changing the answer much.
LIVE_VIDEO_FPS: int = _env_int("LIVE_VIDEO_FPS", 3)

# ── Logging ───────────────────────────────────────────────────────────────────
# When true, every outbound payload is printed in full. Useful while developing;
# in a deployment it writes users' transcribed speech into the server log, so it
# defaults off.
LOG_PAYLOADS: bool = _env_bool("LOG_PAYLOADS", False)
