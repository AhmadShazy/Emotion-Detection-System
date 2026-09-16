"""
api.py — FastAPI Entry Point
============================
Humanoid Assistant V2 API

Serves every analysis mode: text, voice, video upload and the live call.

Frontend static files (UI) are always public — no key needed.
API endpoints require a key, on every machine including this one: callers send
X-API-Key, or ?api_key= for WebSockets, which cannot carry headers.
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

from routers import text, mock, voice, video, stream
from src.core.config import API_KEYS


# ════════════════════════════════════════════════════════════════════════════
# API Key Middleware
# ════════════════════════════════════════════════════════════════════════════

from urllib.parse import parse_qs


def _extract_key(scope) -> str:
    """
    Pulls the caller's key from the header, falling back to the query string.

    The query-string form exists for WebSockets: a browser cannot set headers
    on a WebSocket handshake, so ?api_key= is the only channel available to
    frontend/app.js. .env.example documents both forms.
    """
    for name, value in scope.get("headers", []):
        if name == b"x-api-key":
            return value.decode("latin-1").strip()

    raw = scope.get("query_string", b"")
    values = parse_qs(raw.decode("latin-1")).get("api_key") or []
    return values[0].strip() if values else ""


class APIKeyMiddleware:
    """
    Gates the API on a key, for HTTP requests and WebSocket handshakes alike.

    Always PUBLIC (no key needed):
      - /health, /docs, /redoc, /openapi.json  (monitoring + docs)
      - Frontend static files (/, *.html, *.css, *.js, *.ico etc.)

    Always PROTECTED (key required):
      - /analyze/*   (text, voice, video)
      - /ws/stream   (websocket)

    There is deliberately no exemption for local requests. A bypass keyed on
    where the caller connected from means the thing you test on a laptop is not
    the thing that runs on the server: the auth path — the part most worth
    exercising — would be the one part never exercised. Running locally means
    presenting a key, exactly as a deployed caller does.

    The WebSocket branch is not an afterthought: every check here used to sit
    inside `if scope["type"] == "http"`, so a "websocket" scope fell straight
    through to the app and /ws/stream — the most expensive endpoint in the
    system, and the one with a hard capacity limit — accepted anyone. Two
    anonymous sockets could occupy every slot and lock out real users.
    """
    def __init__(self, app):
        self.app = app

    def _is_authorised(self, scope) -> bool:
        # An empty key set turns authentication off outright. That is a real
        # configuration — a throwaway local demo, or a checkout with no .env —
        # rather than a hidden special case, and the lifespan handler shouts
        # about it at startup so it can never be the accidental state in a
        # deployment.
        if not API_KEYS:
            return True
        return _extract_key(scope) in API_KEYS

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            path    = request.url.path

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

            # ── Apply key check only to API endpoints ─────────────────────────
            if (
                not any(path.startswith(p) for p in open_paths)
                and not is_static_file
                and not self._is_authorised(scope)
            ):
                response = JSONResponse(
                    {"detail": "Invalid or missing API Key."},
                    status_code=403,
                )
                await response(scope, receive, send)
                return

        elif scope["type"] == "websocket":
            # No public WebSocket routes exist, so there is nothing to exempt.
            if not self._is_authorised(scope):
                # Closing before accepting makes the handshake fail, which the
                # browser surfaces as a connection error. 1008 is the "policy
                # violation" code.
                await send({"type": "websocket.close", "code": 1008})
                return

        await self.app(scope, receive, send)


# ════════════════════════════════════════════════════════════════════════════
# Startup / Shutdown lifecycle
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] Humanoid Assistant API initialising...")
    print(f"[STARTUP] API Keys loaded: {len(API_KEYS)}")

    # config.py promises this warning exists so an unauthenticated deployment
    # can never happen quietly. It has to be impossible to miss in a log that
    # also carries a dozen model-loading lines.
    if not API_KEYS:
        print("[STARTUP] " + "!" * 62)
        print("[STARTUP] !!  NO API KEYS SET — EVERY ENDPOINT IS OPEN TO ANYONE  !!")
        print("[STARTUP] !!  Set API_KEYS before exposing this server publicly.  !!")
        print("[STARTUP] " + "!" * 62)

    for sub in ("data/recordings", "data/processed", "data/jobs"):
        os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

    # Clear any working directories orphaned by a previous process that was
    # killed mid-request. Each request cleans up after itself in a finally
    # block, but that cannot survive a SIGKILL, an OOM kill or a reboot.
    from routers.video import sweep_stale_jobs
    swept = sweep_stale_jobs()
    if swept:
        print(f"[STARTUP] Removed {swept} stale job director"
              f"{'y' if swept == 1 else 'ies'} from a previous run.")

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

# Contract sandbox for the LLM team. Loads no models, so it answers instantly
# and stays usable even while the registry is still warming up.
app.include_router(mock.router, prefix="/mock", tags=["Contract Sandbox"])

app.include_router(voice.router,  prefix="/analyze", tags=["Voice Analysis"])
app.include_router(video.router,  prefix="/analyze", tags=["Video Analysis"])
app.include_router(stream.router, tags=["Live Stream"])

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    from src.core.model_registry import registry
    return JSONResponse({
        "status":      "ok",
        "api_version": "2.0.0",
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