from __future__ import annotations

from importlib import resources
from pathlib import Path
import sys


def resource_path(relative: str) -> str:
    """
    Resolve a resource path packaged with geon/resources.
    Falls back to package-relative or CWD paths when needed.
    """
    # PyInstaller onefile bundles unpack to _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "geon" / "resources" / relative
        if candidate.exists():
            return str(candidate.resolve())
    try:
        base = resources.files("geon") / "resources"
        candidate = (base / relative)
        if candidate.exists():
            return str(candidate.resolve())
    except Exception:
        pass
    package_root = Path(__file__).resolve().parents[1]
    candidate = package_root / "resources" / relative
    if candidate.exists():
        return str(candidate.resolve())
    return str((Path("resources") / relative).resolve())
