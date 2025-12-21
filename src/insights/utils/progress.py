from __future__ import annotations

import sys
from threading import Lock
from typing import Callable

ProgressFn = Callable[[str], None]


def make_progress_printer(*, prefix: str = "insights") -> ProgressFn:
    """
    Return a simple progress callback that prints to stderr.

    - Prints only when stderr is a TTY (so pipes/log capture stay clean by default).
    - Thread-safe (best-effort) to avoid interleaving output when used across layers.
    """

    lock = Lock()

    def _progress(message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        if not sys.stderr or not getattr(sys.stderr, "isatty", lambda: False)():
            return
        line = f"{prefix}: {msg}\n"
        with lock:
            try:
                sys.stderr.write(line)
                sys.stderr.flush()
            except Exception:
                # Never fail the main operation due to progress output.
                return

    return _progress


def noop_progress(_: str) -> None:
    return


