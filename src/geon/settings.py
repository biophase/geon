from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    tomllib = None  # type: ignore


DEFAULT_PREFS: Dict[str, Any] = {
    "user_name": "Unnamed User",
    "enable_telemetry": False,
}


def _default_path() -> Path:
    return Path.home() / ".geon_settings.toml"


@dataclass
class Preferences:
    user_name: str = DEFAULT_PREFS["user_name"]
    enable_telemetry: bool = DEFAULT_PREFS["enable_telemetry"]
    path: Path = None  # type: ignore

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = _default_path()

    @classmethod
    def load(cls, path: Path | None = None) -> "Preferences":
        prefs = cls(path=path or _default_path())
        if prefs.path.exists():
            text = prefs.path.read_text(encoding="utf-8")
            data: Dict[str, Any] = {}
            if tomllib is not None:
                try:
                    data = tomllib.loads(text)
                except Exception:
                    data = {}
            else:
                # minimal parse for simple key/value pairs
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if val.lower() in {"true", "false"}:
                        data[key] = val.lower() == "true"
                    elif val.startswith('"') and val.endswith('"'):
                        data[key] = val.strip('"')
                    else:
                        data[key] = val
            prefs.user_name = str(data.get("user_name", prefs.user_name))
            prefs.enable_telemetry = bool(data.get("enable_telemetry", prefs.enable_telemetry))
        return prefs

    def to_toml(self) -> str:
        user = self.user_name.replace('"', '\\"')
        tele = "true" if self.enable_telemetry else "false"
        return f'user_name = "{user}"\nenable_telemetry = {tele}\n'

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_toml(), encoding="utf-8")
