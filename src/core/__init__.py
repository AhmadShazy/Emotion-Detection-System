"""
src/core package.

Applies the UTF-8 console fix on first import so that every entry point —
api.py, main.py, scripts/, or a module run standalone — is protected from
UnicodeEncodeError on Windows cp1252 consoles. See src/core/console.py.
"""

from src.core.console import enable_utf8_console

enable_utf8_console()
