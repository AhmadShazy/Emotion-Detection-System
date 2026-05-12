"""
src/core/config.py
==================
Central configuration loader.
Reads environment variables from .env file.
All other modules import from here — never read os.environ directly.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ── Mode ──────────────────────────────────────────────────────────────────────
TEXT_ONLY_MODE: bool = os.getenv("TEXT_ONLY_MODE", "false").lower() == "true"

# ── Auth ──────────────────────────────────────────────────────────────────────
# Supports multiple API keys separated by commas in .env
# To add a new teammate: add their key to API_KEYS in .env on the server
# Example: API_KEYS=humanoid_ahmad_2024,humanoid_ali_2024,humanoid_sara_2024
_raw_keys = os.getenv("API_KEYS", os.getenv("API_KEY", ""))
API_KEYS: set = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# ── LLM Forwarding ────────────────────────────────────────────────────────────
LLM_ENDPOINT_URL: str = os.getenv("LLM_ENDPOINT_URL", "")

# ── Local Development ─────────────────────────────────────────────────────────
ALLOW_LOCALHOST: bool = os.getenv("ALLOW_LOCALHOST", "false").lower() == "true"