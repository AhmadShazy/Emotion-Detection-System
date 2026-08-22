"""
src/core/console.py
===================
Forces UTF-8 on stdout/stderr so the emoji and box-drawing characters used
throughout this project's logging can never crash the process.

Why this exists
---------------
On Windows the default console encoding is cp1252, which cannot represent
characters like '═', '✅', '⚠️' or '🎉'. Because those appear in nearly every
log line in this codebase, a plain print() raises:

    UnicodeEncodeError: 'charmap' codec can't encode characters ...

That is not cosmetic — it is an unhandled exception. It was aborting
registry.load_all() partway through startup, so the API could not boot at
all on a default Windows console.

Rather than stripping the emoji out of hundreds of print() calls, we widen
the streams once at process start. errors="replace" is a safety net: if a
stream genuinely cannot do UTF-8, unencodable characters degrade to '?'
instead of raising.

Usage
-----
Called automatically by src/core/__init__.py, so any import of src.core.*
is covered. Entry points (api.py, main.py) also call it explicitly as the
very first statement, which documents the dependency and covers the case
where they print before importing anything from src.core.
"""

import sys

_configured = False


def enable_utf8_console() -> None:
    """
    Reconfigures sys.stdout / sys.stderr to UTF-8.

    Idempotent — safe to call from every entry point.
    Never raises: a stream that cannot be reconfigured is left alone.
    """
    global _configured
    if _configured:
        return

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)

        # stdout/stderr can be None (pythonw.exe) or a replacement object
        # without .reconfigure() (pytest capture, some IDE consoles).
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Never let logging setup break the program it is meant to serve.
            pass

    _configured = True
