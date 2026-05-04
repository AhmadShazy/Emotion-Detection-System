"""
api.py — FastAPI Entry Point
============================
Humanoid Assistant V2 API

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000          → Frontend UI
    http://localhost:8000/docs     → Swagger API docs
    http://localhost:8000/redoc    → ReDoc API docs

Endpoints registered:
    POST  /analyze/text                  → Text emotion analysis
    POST  /analyze/voice                 → Voice (.wav) analysis
    POST  /analyze/multimodal/start      → Start server-side recording
    POST  /analyze/multimodal/stop       → Stop + analyze recording
    WS    /ws/stream                     → Live multimodal stream
    GET   /                              → Frontend (served from frontend/)
    GET   /health                        → Health check + model status
"""

import sys
import os
import asyncio

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
    Runs once on server start.

    ALL models are loaded here via ModelRegistry.load_all() before yield.
    FastAPI guarantees no request is handled before yield completes.
    This means every endpoint is guaranteed to have all models ready
    from the very first request — no lazy loading, no runtime downloads.

    asyncio.to_thread() is used because load_all() is blocking (~17s of
    CPU/disk work). Running it in a thread pool keeps the event loop free.
    """
    print("\n[STARTUP] Humanoid Assistant API initialising...")

    # Ensure required data directories exist
    for sub in ("data/recordings", "data/processed"):
        os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

    # Load ALL models before serving any request.
    # asyncio.to_thread() runs the blocking load_all() in a thread pool
    # so the event loop is not blocked during the ~17s startup.
    from src.core.model_registry import registry
    await asyncio.to_thread(registry.load_all)

    print("[STARTUP] ✅ API ready — all models loaded.\n")
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
# ════════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
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
# Health check
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    """
    Returns 200 OK with server info and model load status.
    Check the 'models' field to verify all four models loaded successfully.
    """
    from src.core.model_registry import registry
    return JSONResponse({
        "status":      "ok",
        "api_version": "2.0.0",
        "project":     "Humanoid Assistant",
        "models":      registry.status(),
    })


# ════════════════════════════════════════════════════════════════════════════
# Static Frontend
# Mount LAST — catches everything not matched by routers above.
# Logs clearly so 404 issues are immediately visible in the console.
# ════════════════════════════════════════════════════════════════════════════

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
    print(f"[STARTUP]    PROJECT_ROOT resolved to: {PROJECT_ROOT}")
    print(f"[STARTUP]    GET / will return 404 until frontend/ exists.")