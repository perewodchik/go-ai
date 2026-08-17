"""
console.py — make stdout able to carry this project's output on Windows.

Every progress line this project prints is decorated with an emoji (🎯, 🧠, 🏳️,
🚨). On Linux and macOS stdout is UTF-8 and that is free. On Windows a console
inherits the system ANSI code page — cp1252 for a Western install — and printing
a character it cannot encode does not degrade, it raises UnicodeEncodeError.

That is not a cosmetic failure. `run_server.py` prints its banner BEFORE
`socketio.run()`, so the exception happens during startup and the server never
binds its port: the whole application is unreachable because of a target emoji.
`run_training.py` fails the same way on its first progress line.

Called once from each entry point, before anything prints.
"""

import sys


def use_utf8_console() -> None:
    """
    Switch stdout/stderr to UTF-8, replacing anything the terminal cannot show.

    `errors="replace"` rather than "strict" is the point: a console font
    without emoji coverage should print a placeholder, never take the process
    down. On a stream that is already UTF-8 this is a no-op, so it is safe to
    call unconditionally.

    Both streams are wrapped because tracebacks go to stderr, and a crash
    report that itself crashes on an emoji is the worst version of this bug.
    """
    for stream in (sys.stdout, sys.stderr):
        # A redirected or wrapped stream may not be reconfigurable (pytest's
        # capture, a pipe from another process). Not being able to set it is
        # never a reason to fail.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
