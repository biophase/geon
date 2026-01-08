from __future__ import annotations

from importlib import resources
from pathlib import Path


def resource_path(relative: str) -> str:
    """
    Resolve a resource path packaged with geon/resources.
    Falls back to a relative path if importlib.resources is unavailable.
    """
    try:
        base = resources.files("geon") / "resources"
        candidate = (base / relative)
        if candidate.exists():
            return str(candidate.resolve())
    except Exception:
        pass
    return str((Path("resources") / relative).resolve())
