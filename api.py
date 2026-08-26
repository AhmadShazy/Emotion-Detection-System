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
    from routers import voice, video, stream


# ════════════════════════════════════════════════════════════════════════════
# API Key Middleware
# ════════════════════════════════════════════════════════════════════════════

from urllib.parse import parse_qs

from src.core.config import ALLOW_LOCALHOST

# Loopback addresses a real TCP peer can actually present. "localhost" and
# "0.0.0.0" are not in this set on purpose: a peer address is an IP, never a
# hostname, and 0.0.0.0 is a bind address rather than something a client can
# connect from.
_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _peer_is_loopback(scope) -> bool:
    """
    True when the request really arrived over the loopback interface.

    Reads scope["client"], the actual socket peer. It deliberately does NOT
    read the Host header, which any client sets to whatever it likes — sending
    "Host: localhost" to a public server would otherwise skip the key check
    entirely. src/core/config.py documents this guarantee; this is the code
    that keeps it.

    ⚠️  Behind a reverse proxy (Cloud Run, Render, Railway) the socket peer is
    the PROXY, and on some platforms that address is on loopback. Turning
    ALLOW_LOCALHOST on in such a deployment would therefore expose the whole
    API, not just local traffic. Leave it false anywhere but a laptop, and if
    that ever changes, run uvicorn with --proxy-headers and read
    X-Forwarded-For instead.
    """
    if not ALLOW_LOCALHOST:
        return False
    client = scope.get("client")
    if not client:
        return False
    return str(client[0]) in _LOOPBACK


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
      - Requests whose real socket peer is loopback, when ALLOW_LOCALHOST=true

    Always PROTECTED (key required):
      - /analyze/*   (text, voice, video)
      - /ws/stream   (websocket)

    The WebSocket branch is not an afterthought: every check here used to sit
    inside `if scope["type"] == "http"`, so a "websocket" scope fell straight
    through to the app and /ws/stream — the most expensive endpoint in the
    system, and the one with a hard capacity limit — accepted anyone. Two
    anonymous sockets could occupy every slot and lock out real users.
    """
    def __init__(self, app):
        self.app = app

    def _is_authorised(self, scope) -> bool:
        # An empty key set disables auth deliberately (local dev, text-only
        # demos). The lifespan handler shouts about it at startup so it can
        # never be the accidental state in a deployment.
        if not API_KEYS:
            return True
        if _peer_is_loopback(scope):
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
    print(f"[STARTUP] Mode: {'TEXT ONLY' if TEXT_ONLY_MODE else 'FULL'}")
    print(f"[STARTUP] API Keys loaded: {len(API_KEYS)}")

    # config.py promises this warning exists so an unauthenticated deployment
    # can never happen quietly. It has to be impossible to miss in a log that
    # also carries a dozen model-loading lines.
    if not API_KEYS:
        print("[STARTUP] " + "!" * 62)
        print("[STARTUP] !!  NO API KEYS SET — EVERY ENDPOINT IS OPEN TO ANYONE  !!")
        print("[STARTUP] !!  Set API_KEYS before exposing this server publicly.  !!")
        print("[STARTUP] " + "!" * 62)
    if ALLOW_LOCALHOST:
        print("[STARTUP] ⚠️  ALLOW_LOCALHOST=true — loopback callers skip the key "
              "check. Correct on a laptop, wrong behind a cloud proxy.")

    for sub in ("data/recordings", "data/processed", "data/jobs"):
        os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

    # Clear any working directories orphaned by a previous process that was
    # killed mid-request. Each request cleans up after itself in a finally
    # block, but that cannot survive a SIGKILL, an OOM kill or a reboot.
    if not TEXT_ONLY_MODE:
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

# Contract sandbox for the LLM team. Registered in BOTH modes and loads no
# models, so it stays available on a small text-only deployment.
app.include_router(mock.router, prefix="/mock", tags=["Contract Sandbox"])

if not TEXT_ONLY_MODE:
    app.include_router(voice.router,      prefix="/analyze", tags=["Voice Analysis"])
    app.include_router(video.router,      prefix="/analyze", tags=["Video Analysis"])
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