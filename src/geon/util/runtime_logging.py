from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import TextIO


_LOG_STREAM: TextIO | None = None


def _log_path(app_name: str) -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata)
    else:
        base = Path.home()
    log_dir = base / app_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{app_name}.log"


def _write_line(stream: TextIO, message: str) -> None:
    stream.write(message.rstrip() + "\n")
    stream.flush()


def _redirect_stdio(stream: TextIO) -> None:
    sys.stdout = stream
    sys.stderr = stream
    try:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    except OSError:
        pass


def _install_exception_hooks(stream: TextIO) -> None:
    def _sys_hook(exc_type, exc, tb) -> None:
        _write_line(stream, "\n[geon] Unhandled exception")
        traceback.print_exception(exc_type, exc, tb, file=stream)
        stream.flush()

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        _write_line(stream, f"\n[geon] Unhandled thread exception in {args.thread.name!r}")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=stream)
        stream.flush()

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


def configure_runtime_logging(app_name: str = "geon") -> Path | None:
    global _LOG_STREAM
    if _LOG_STREAM is not None:
        return None

    needs_redirect = getattr(sys, "frozen", False) or sys.stdout is None or sys.stderr is None
    if not needs_redirect:
        return None

    path = _log_path(app_name)
    stream = path.open("a", encoding="utf-8", buffering=1, errors="replace")
    _LOG_STREAM = stream
    _redirect_stdio(stream)
    _install_exception_hooks(stream)
    try:
        faulthandler.enable(stream, all_threads=True)
    except Exception:
        pass
    _write_line(stream, f"[geon] Logging redirected to {path}")
    _write_line(stream, f"[geon] Python executable: {sys.executable}")
    return path
