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
    """Middleware with one known key and the localhost bypass switched off."""
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", False)
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
    """Deliberate for local dev and text-only demos; startup warns loudly."""
    monkeypatch.setattr(api, "API_KEYS", set())
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", False)
    gate = api.APIKeyMiddleware(app=None)
    assert gate._is_authorised(http_scope()) is True
    assert gate._is_authorised(ws_scope()) is True


# ── The localhost bypass ──────────────────────────────────────────────────────

def test_host_header_cannot_fake_a_local_request(monkeypatch):
    """
    The bypass reads the real socket peer, never the Host header.

    It used to derive the host from request.url.hostname, which Starlette takes
    from the Host header — a value the caller controls. Sending
    "Host: localhost" to a public server skipped the key check entirely, while
    src/core/config.py documented the opposite guarantee.
    """
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", True)
    gate = api.APIKeyMiddleware(app=None)

    spoofed = http_scope(
        headers=[(b"host", b"localhost")],
        client=("203.0.113.7", 51234),      # a real remote peer
    )
    assert gate._is_authorised(spoofed) is False, (
        "a Host header of 'localhost' must not grant the local bypass"
    )


def test_real_loopback_peer_skips_the_key_check(monkeypatch):
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", True)
    gate = api.APIKeyMiddleware(app=None)

    for addr in ("127.0.0.1", "::1"):
        assert gate._is_authorised(http_scope(client=(addr, 51234))) is True


def test_bypass_is_off_unless_explicitly_enabled(monkeypatch):
    """ALLOW_LOCALHOST defaults false, so even loopback needs a key."""
    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", False)
    gate = api.APIKeyMiddleware(app=None)
    assert gate._is_authorised(http_scope(client=("127.0.0.1", 51234))) is False


# ── End to end ────────────────────────────────────────────────────────────────

def test_websocket_handshake_is_refused_without_a_key(monkeypatch):
    """
    Drives the real ASGI stack. No lifespan is started, so no models load —
    the rejection happens in middleware, before any route code runs.
    """
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(api, "API_KEYS", {REAL_KEY})
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", False)

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
    monkeypatch.setattr(api, "ALLOW_LOCALHOST", False)

    client = TestClient(api.app)
    with client.websocket_connect(f"/ws/stream?api_key={REAL_KEY}") as ws:
        hello = ws.receive_json()
        assert hello["code"] == "CONNECTED"
        ws.send_json({"type": "stop"})
