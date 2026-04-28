"""Tee print() output to stdout AND a file, line-buffered.

The default Python stdio buffering eats progress bars when piped to `tail`
or to a wakeup task. This helper replaces stdout/stderr with a
LineFlushedTee so each `print(..., flush=True)` (and even un-flushed prints
once a newline arrives) lands on disk immediately.

Usage:
    from streaming_log import setup_logging
    log_path = setup_logging("smoke_run.log")     # also prints to terminal
    print("hello")                                # appears in both
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


class _LineFlushedTee:
    """Write to two underlying streams; flush after every line break."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                if "\n" in data:
                    s.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


def setup_logging(log_path: str | Path, append: bool = False) -> Path:
    """Tee sys.stdout / sys.stderr to `log_path`, line-buffered.

    Returns the resolved absolute path."""
    p = Path(log_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    # buffering=1 → line-buffered (POSIX). On Windows it falls back to block
    # buffering for binary files, but we open as text with explicit newline
    # which preserves the line-flush contract.
    f = open(p, mode, encoding="utf-8", buffering=1, newline="\n")

    sys.stdout = _LineFlushedTee(sys.__stdout__, f)
    sys.stderr = _LineFlushedTee(sys.__stderr__, f)

    # Also disable Python-level stdout buffering so `print` without
    # flush=True still shows up reasonably fast.
    try:
        sys.__stdout__.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(f"[log] streaming → {p}", flush=True)
    return p
