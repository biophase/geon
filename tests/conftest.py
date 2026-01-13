from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    if os.getenv("GEON_USE_INSTALLED") == "1":
        return
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


_ensure_src_on_path()
