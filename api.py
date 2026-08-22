"""
api.py — FastAPI Entry Point
============================
Humanoid Assistant V2 API

TEXT_ONLY_MODE=true  → only /analyze/text + /health exposed
TEXT_ONLY_MODE=false → all endpoints exposed

Localhost requests bypass API key check for local development.
Frontend static files (UI) are always public — no key needed.
API endpoints require X-API-Key header.
"""

import sys
import os
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Widen stdout/stderr to UTF-8 before anything logs. Startup banners and most
# log lines contain emoji / box-drawing characters, which raise
# UnicodeEncodeError on a default Windows cp1252 console and abort startup.
from src.core.console import enable_utf8_console
enable_utf8_console()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from routers import text, mock
from src.core.config import TEXT_ONLY_MODE, API_KEYS

# ── Conditionally import disabled routers ─────────────────────────────────────
if not TEXT_ONLY_MODE:
    from routers import voice, multimodal, stream


# ════════════════════════════════════════════════════════════════════════════
# API Key Middleware
# ════════════════════════════════════════════════════════════════════════════

from src.core.config import ALLOW_LOCALHOST
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"} if ALLOW_LOCALHOST else set()


class APIKeyMiddleware:
    """
    Checks X-API-Key header on API endpoints.

    Always PUBLIC (no key needed):
      - /health, /docs, /redoc, /openapi.json  (monitoring + docs)
      - Frontend static files (/, *.html, *.css, *.js, *.ico etc.)
      - Requests from localhost / 127.0.0.1 (local dev)

    Always PROTECTED (key required):
      - /analyze/*   (text, voice, multimodal)
      - /ws/stream   (websocket)
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            path    = request.url.path
            host    = request.url.hostname or ""

            # ── Always public — monitoring and docs ───────────────────────────
            open_paths = ["/health", "/docs", "/redoc", "/openapi.json"]

            # ── Always public — frontend static files ─────────────────────────
            # Covers: /, /index.html, /app.js, /style.css, /favicon.ico etc.
            last_segment = path.split("/")[-1]
            is_static_file = (
                path == "/"
                or path.startswith("/static")
                or ("." in last_segment and not path.startswith("/analyze"))
            )

            # ── Local development — skip key check ────────────────────────────
            is_local = host in _LOCAL_HOSTS

            # ── Apply key check only to API endpoints ─────────────────────────
            if (
                not any(path.startswith(p) for p in open_paths)
                and not is_static_file
                and not is_local
            ):
                key = request.headers.get("X-API-Key", "")
                if API_KEYS and key not in API_KEYS:
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
    print(f"[STARTUP] API Keys loaded: {len(API_KEYS)}")

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

# Contract sandbox for the LLM team. Registered in BOTH modes and loads no
# models, so it stays available on a small text-only deployment.
app.include_router(mock.router, prefix="/mock", tags=["Contract Sandbox"])

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