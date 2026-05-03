"""
api.py — FastAPI Entry Point
============================
Humanoid Assistant V2 API

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000          → Frontend UI
    http://localhost:8000/docs     → Interactive API docs (Swagger)
    http://localhost:8000/redoc    → ReDoc API docs

Endpoints registered:
    POST  /analyze/text                  → Text emotion analysis
    POST  /analyze/voice                 → Voice (.wav) analysis
    POST  /analyze/multimodal/start      → Start server-side recording
    POST  /analyze/multimodal/stop       → Stop + analyze recording
    WS    /ws/stream                     → Live multimodal stream
    GET   /                              → Frontend (served from frontend/)
    GET   /health                        → Health check
"""

import sys
import os

# Ensure the project root is always on the path regardless of
# where uvicorn is launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# ── Router imports ────────────────────────────────────────────────────────────
from routers import text, voice, multimodal, stream


# ════════════════════════════════════════════════════════════════════════════
# Startup / Shutdown lifecycle
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on server start — preloads the text emotion model (RoBERTa)
    so the first /analyze/text request isn't slow.
    Everything else (Whisper, SER) loads lazily on first use.
    """
    print("\n[STARTUP] Humanoid Assistant API initialising...")

    try:
        from src.text_emotion.analysis import load_emotion_model
        load_emotion_model()
        print("[STARTUP] ✅ Text emotion model (RoBERTa) loaded.")
    except Exception as e:
        print(f"[STARTUP] ⚠️  Could not preload text emotion model: {e}")

    # Ensure required data directories exist
    for sub in ("data/recordings", "data/processed"):
        os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

    print("[STARTUP] ✅ API ready.\n")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("\n[SHUTDOWN] Humanoid Assistant API shutting down...")


# ════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Humanoid Assistant API",
    description=(
        "Multimodal emotion analysis API.\n\n"
        "Supports **Text**, **Voice**, **Multimodal** (camera + mic), and "
        "**Live Streaming** (WebSocket) modes. "
        "All modes return the unified V2 emotion payload."
    ),
    version="2.0.0",
    contact={
        "name": "Humanoid Assistant Project",
    },
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════════════════════
# CORS Middleware
# Allow all localhost origins so the frontend (served on port 8000) and any
# dev tools (Postman, browser console, etc.) can hit the API freely.
# ════════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",   # in case of a separate dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# Routers
# ════════════════════════════════════════════════════════════════════════════

app.include_router(text.router,       prefix="/analyze", tags=["Text Analysis"])
app.include_router(voice.router,      prefix="/analyze", tags=["Voice Analysis"])
app.include_router(multimodal.router, prefix="/analyze", tags=["Multimodal Analysis"])
app.include_router(stream.router,     tags=["Live Stream"])


# ════════════════════════════════════════════════════════════════════════════
# Health check  (useful for monitoring / Docker readiness probes)
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    """Returns 200 OK with basic server info when the API is running."""
    return JSONResponse({
        "status": "ok",
        "api_version": "2.0.0",
        "project": "Humanoid Assistant",
    })


# ════════════════════════════════════════════════════════════════════════════
# Static Frontend
# Mount LAST — catches everything not matched by the routers above.
# Serves frontend/index.html at GET /
# ════════════════════════════════════════════════════════════════════════════

_frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
