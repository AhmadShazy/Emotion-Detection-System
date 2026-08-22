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


# ── Mode ──────────────────────────────────────────────────────────────────────
# true  → only /analyze/text and /mock/* are served, and only RoBERTa is loaded.
#         Suitable for a small hosted instance the LLM team can build against.
# false → every mode, all four models. Needs a microphone, a camera and OpenFace.
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

# ── Local Development ─────────────────────────────────────────────────────────
# When true, requests whose real TCP peer is a loopback address skip the key
# check, so local development needs no key. Enforced against the actual socket
# peer, never the Host header, which a client controls.
#
# Leave this FALSE in any deployment.
ALLOW_LOCALHOST: bool = _env_bool("ALLOW_LOCALHOST", False)
