"""
tests/test_auth.py
==================
Tests for the API key gate in api.py.

The WebSocket half is the reason this file exists. Every check in
APIKeyMiddleware used to sit inside `if scope["type"] == "http"`, so a
"websocket" scope fell straight through and /ws/stream — the most expensive
endpoint in the system, with a hard capacity limit — accepted anyone who knew
the URL. The live-stream test passed a key the server never read, so it would
have passed identically with authentication deleted.

Most of these drive the middleware's own helpers with hand-built ASGI scopes,
so they need no server and no models and run in milliseconds.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import api


REAL_KEY = "hma_test_key_aaaaaaaaaaaaaaaaaaaaaaaa"


def http_scope(headers=None, query=b"", client=("203.0.113.7", 51234)):
    """A public-internet HTTP request unless told otherwise."""
    return {
        "type": "http",
        "path": "/analyze/text",
        "headers": headers or [],
        "query_string": query,
        "client": client,
    }


def ws_scope(headers=None, query=b"", client=("203.0.113.7", 51234)):
    return {
        "type": "websocket",
        "path": "/ws/stream",
        "headers": headers or [],
        "query_string": query,
        "client": client,
    }


@pytest.fixture
def gate(monkeypatch):
    """Middleware with one known key configured."""
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    return api.APIKeyMiddleware(app=None)


# ── Where the key is read from ────────────────────────────────────────────────

def test_key_is_read_from_the_header():
    scope = http_scope(headers=[(b"x-api-key", REAL_KEY.encode())])
    assert api._extract_key(scope) == REAL_KEY


def test_key_is_read_from_the_query_string():
    """
    A browser cannot set headers on a WebSocket handshake, so ?api_key= is the
    only channel frontend/app.js has. .env.example documents both forms.
    """
    scope = ws_scope(query=f"api_key={REAL_KEY}".encode())
    assert api._extract_key(scope) == REAL_KEY


def test_missing_key_reads_as_empty():
    assert api._extract_key(http_scope()) == ""
    assert api._extract_key(ws_scope(query=b"session_id=abc")) == ""


# ── The gate itself ───────────────────────────────────────────────────────────

def test_correct_key_is_accepted(gate):
    assert gate._is_authorised(
        http_scope(headers=[(b"x-api-key", REAL_KEY.encode())])
    ) is True


def test_wrong_key_is_rejected(gate):
    assert gate._is_authorised(
        http_scope(headers=[(b"x-api-key", b"not-the-key")])
    ) is False


def test_absent_key_is_rejected(gate):
    assert gate._is_authorised(http_scope()) is False


def test_websocket_scope_is_gated_exactly_like_http(gate):
    """
    The regression this file was written for. A websocket scope must go through
    the same decision as an http one — not past it.
    """
    assert gate._is_authorised(
        ws_scope(query=f"api_key={REAL_KEY}".encode())
    ) is True
    assert gate._is_authorised(ws_scope(query=b"api_key=wrong")) is False
    assert gate._is_authorised(ws_scope()) is False


def test_empty_key_set_disables_the_gate(monkeypatch):
    """
    A real configuration rather than a hidden special case: a throwaway local
    demo, or a checkout with no .env. Startup prints a loud banner when it
    happens.
    """
    monkeypatch.setattr(api, "API_KEYS", set())
    gate = api.APIKeyMiddleware(app=None)
    assert gate._is_authorised(http_scope()) is True
    assert gate._is_authorised(ws_scope()) is True


# ── No exemption for local callers ────────────────────────────────────────────

def test_a_local_caller_still_needs_a_key(gate):
    """
    Requests from this machine take the same path as requests from anywhere
    else.

    There used to be an ALLOW_LOCALHOST bypass that skipped the key check for
    loopback callers. It meant the auth path — the part most worth exercising —
    was the one part local testing never touched, and behind a cloud proxy the
    socket peer is the proxy, sometimes itself on loopback, so enabling it in a
    deployment would have opened everything rather than nothing.
    """
    for addr in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        assert gate._is_authorised(http_scope(client=(addr, 51234))) is False
        assert gate._is_authorised(ws_scope(client=(addr, 51234))) is False

    # ...and the same caller succeeds once it presents the key.
    assert gate._is_authorised(
        http_scope(headers=[(b"x-api-key", REAL_KEY.encode())],
                   client=("127.0.0.1", 51234))
    ) is True


def test_no_header_can_talk_the_gate_into_trusting_a_request(gate):
    """
    The decision reads the key and nothing else.

    An earlier version derived the caller's host from request.url.hostname,
    which Starlette takes from the Host header — a value the caller sets.
    Sending "Host: localhost" to a public server skipped the check entirely.
    Nothing header-derived should be able to stand in for a key.
    """
    for spoof in (
        [(b"host", b"localhost")],
        [(b"host", b"127.0.0.1")],
        [(b"x-forwarded-for", b"127.0.0.1")],
        [(b"x-real-ip", b"127.0.0.1")],
    ):
        assert gate._is_authorised(
            http_scope(headers=spoof, client=("203.0.113.7", 51234))
        ) is False, f"{spoof} must not grant access"


def test_the_bypass_setting_is_really_gone():
    """
    Guards the removal itself. A stale ALLOW_LOCALHOST left in config would
    read as live configuration to anyone setting up the project, and someone
    would eventually set it expecting it to do something.
    """
    import src.core.config as config
    assert not hasattr(config, "ALLOW_LOCALHOST")
    assert not hasattr(api, "ALLOW_LOCALHOST")
    assert not hasattr(api, "_peer_is_loopback")


def test_text_only_mode_is_really_gone():
    """
    Guards the other removal, for the same reason.

    The server once had a reduced mode serving only /analyze/text, built for a
    free tier that could not host the full stack. That plan was dropped and the
    flag became a second code path nothing exercised: half the routes carried a
    disabled twin, the registry had two loading strategies, and the frontend
    asked /health which of them it was talking to. Every mode is always
    available now.

    A stale TEXT_ONLY_MODE left anywhere would read as live configuration, and
    someone would eventually set it expecting it to do something.
    """
    import src.core.config as config
    import src.core.model_registry as registry_mod

    assert not hasattr(config, "TEXT_ONLY_MODE")
    assert not hasattr(api, "TEXT_ONLY_MODE")
    assert not hasattr(registry_mod, "TEXT_ONLY_MODE")

    # Every analysis route must be registered unconditionally.
    paths = {r.path for r in api.app.routes}
    for required in ("/analyze/text", "/analyze/voice", "/analyze/video", "/ws/stream"):
        assert required in paths, f"{required} is not registered"


# ── End to end ────────────────────────────────────────────────────────────────

def test_websocket_handshake_is_refused_without_a_key(monkeypatch):
    """
    Drives the real ASGI stack. No lifespan is started, so no models load —
    the rejection happens in middleware, before any route code runs.
    """
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})

    client = TestClient(api.app)
    # Deliberately narrow. `pytest.raises(Exception)` would also pass if the
    # test broke for an unrelated reason, which is how a regression test ends
    # up guarding nothing.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream?api_key=wrong") as ws:
            ws.receive_json()


def test_websocket_handshake_succeeds_with_a_valid_key(monkeypatch):
    """
    The other half of the pair. Without this, a middleware that rejected every
    socket unconditionally would pass the test above and still be broken.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})

    client = TestClient(api.app)
    with client.websocket_connect(f"/ws/stream?api_key={REAL_KEY}") as ws:
        hello = ws.receive_json()
        assert hello["code"] == "CONNECTED"
        ws.send_json({"type": "stop"})
