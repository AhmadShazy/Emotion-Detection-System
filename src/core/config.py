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
API_KEY: str = os.getenv("API_KEY", "")

# ── LLM Forwarding ────────────────────────────────────────────────────────────
LLM_ENDPOINT_URL: str = os.getenv("LLM_ENDPOINT_URL", "")