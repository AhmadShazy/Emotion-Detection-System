"""
api.py — FastAPI Entry Point
============================
Humanoid Assistant V2 API

TEXT_ONLY_MODE=true  → only /analyze/text + /health exposed
TEXT_ONLY_MODE=false → all endpoints exposed

Localhost requests bypass API key check for local development.
"""

import sys
import os
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from routers import text
from src.core.config import TEXT_ONLY_MODE, API_KEY

# ── Conditionally import disabled routers ─────────────────────────────────────
if not TEXT_ONLY_MODE:
    from routers import voice, multimodal, stream


# ════════════════════════════════════════════════════════════════════════════
# API Key Middleware
# ════════════════════════════════════════════════════════════════════════════

# Hosts that bypass API key check — controlled by ALLOW_LOCALHOST in .env
from src.core.config import ALLOW_LOCALHOST
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"} if ALLOW_LOCALHOST else set()

class APIKeyMiddleware:
    """
    Checks X-API-Key header on every request.

    Skips check for:
      - /health, /docs, /redoc, /openapi.json  (monitoring + docs)
      - requests from localhost / 127.0.0.1    (local dev access)
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            path    = request.url.path
            host    = request.url.hostname or ""

            # Always public — monitoring and docs
            open_paths = ["/health", "/docs", "/redoc", "/openapi.json"]

            # Local development — skip key check
            is_local = host in _LOCAL_HOSTS

            if not any(path.startswith(p) for p in open_paths) and not is_local:
                key = request.headers.get("X-API-Key", "")
                if API_KEY and key != API_KEY:
                    response = JSONResponse(
                        {"detail": "Invalid or missing API Key."},
                        status_code=403,
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)


# ════════════════════════════════════════════════════════════════════════════
# Startup / Shutdown lifecycle
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] Humanoid Assistant API initialising...")
    print(f"[STARTUP] Mode: {'TEXT ONLY' if TEXT_ONLY_MODE else 'FULL'}")

    for sub in ("data/recordings", "data/processed"):
        os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

    from src.core.model_registry import registry
    await asyncio.to_thread(registry.load_all)

    print("[STARTUP] ✅ API ready.\n")
    yield

    print("\n[SHUTDOWN] Humanoid Assistant API shutting down...")


# ════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Humanoid Assistant API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Add API Key Middleware ─────────────────────────────────────────────────────
app.add_middleware(APIKeyMiddleware)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(text.router, prefix="/analyze", tags=["Text Analysis"])

if not TEXT_ONLY_MODE:
    app.include_router(voice.router,      prefix="/analyze", tags=["Voice Analysis"])
    app.include_router(multimodal.router, prefix="/analyze", tags=["Multimodal Analysis"])
    app.include_router(stream.router,     tags=["Live Stream"])

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    from src.core.model_registry import registry
    return JSONResponse({
        "status":      "ok",
        "api_version": "2.0.0",
        "mode":        "text_only" if TEXT_ONLY_MODE else "full",
        "models":      registry.status(),
    })

# ── Static Frontend ───────────────────────────────────────────────────────────
_frontend_dir = os.path.join(PROJECT_ROOT, "frontend")

if os.path.isdir(_frontend_dir):
    print(f"[STARTUP] Mounting frontend from: {_frontend_dir}")
    app.mount(
        "/",
        StaticFiles(directory=_frontend_dir, html=True),
        name="frontend",
    )
else:
    print(f"[STARTUP] ⚠️  Frontend directory not found at: {_frontend_dir}")