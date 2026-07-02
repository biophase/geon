from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    tomllib = None  # type: ignore


DEFAULT_PREFS: Dict[str, Any] = {
    "user_name": "Unnamed User",
    "enable_telemetry": False,
    "camera_sensitivity": 10.0,
    "cell_complex_size_mode": "screen",
    "cell_complex_screen_size_px": 12.0,
    "cell_complex_world_size": 0.1,
    "cell_complex_edge_width": 1.0,
    "cell_complex_reference_label_text_size_px": 14.0,
    "cell_complex_default_color": [204, 204, 204],
    "selection_color": [255, 128, 0],
    "viewport_text_color": [255, 255, 255],
}

REGION_GROWING_PREFIX = "region_growing__"
PLANE_RANSAC_PREFIX = "plane_ransac__"
SUPERPOINTS_PREFIX = "superpoints__"
REGION_MERGE_PREFIX = "region_merge__"

_TOOL_PREFIXES: Dict[str, str] = {
    "region_growing": REGION_GROWING_PREFIX,
    "plane_ransac": PLANE_RANSAC_PREFIX,
    "superpoints": SUPERPOINTS_PREFIX,
    "region_merge": REGION_MERGE_PREFIX,
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items: list[Any] = []
        current: list[str] = []
        in_string = False
        escaped = False
        for ch in inner:
            if escaped:
                current.append(ch)
                escaped = False
                continue
            if ch == "\\" and in_string:
                current.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                current.append(ch)
                continue
            if ch == "," and not in_string:
                token = "".join(current).strip()
                if token:
                    items.append(_parse_scalar(token))
                current = []
                continue
            current.append(ch)
        token = "".join(current).strip()
        if token:
            items.append(_parse_scalar(token))
        return items
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith('"') and value.endswith('"'):
        return value.strip('"')
    if value.lower() in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _toml_scalar(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _default_path() -> Path:
    return Path.home() / ".geon_settings.toml"


def _tool_settings_from_data(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in data.items()
        if key.startswith(prefix)
    }


def _is_supported_setting_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(v, (str, bool, int, float)) for v in value)
    return False


def _sanitize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): (list(value) if isinstance(value, tuple) else value)
        for key, value in settings.items()
        if _is_supported_setting_value(value)
    }


@dataclass
class Preferences:
    user_name: str = DEFAULT_PREFS["user_name"]
    enable_telemetry: bool = DEFAULT_PREFS["enable_telemetry"]
    camera_sensitivity: float = DEFAULT_PREFS["camera_sensitivity"]
    cell_complex_size_mode: str = DEFAULT_PREFS["cell_complex_size_mode"]
    cell_complex_screen_size_px: float = DEFAULT_PREFS["cell_complex_screen_size_px"]
    cell_complex_world_size: float = DEFAULT_PREFS["cell_complex_world_size"]
    cell_complex_edge_width: float = DEFAULT_PREFS["cell_complex_edge_width"]
    cell_complex_reference_label_text_size_px: float = DEFAULT_PREFS[
        "cell_complex_reference_label_text_size_px"
    ]
    cell_complex_default_color: list[int] = field(
        default_factory=lambda: list(DEFAULT_PREFS["cell_complex_default_color"])
    )
    selection_color: list[int] = field(
        default_factory=lambda: list(DEFAULT_PREFS["selection_color"])
    )
    viewport_text_color: list[int] = field(
        default_factory=lambda: list(DEFAULT_PREFS["viewport_text_color"])
    )
    region_growing_settings: Dict[str, Any] = field(default_factory=dict)
    plane_ransac_settings: Dict[str, Any] = field(default_factory=dict)
    superpoints_settings: Dict[str, Any] = field(default_factory=dict)
    region_merge_settings: Dict[str, Any] = field(default_factory=dict)
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
                    data[key] = _parse_scalar(val)
            prefs.user_name = str(data.get("user_name", prefs.user_name))
            prefs.enable_telemetry = bool(data.get("enable_telemetry", prefs.enable_telemetry))
            cam_val = data.get("camera_sensitivity", prefs.camera_sensitivity)
            try:
                prefs.camera_sensitivity = float(cam_val)
            except (TypeError, ValueError):
                prefs.camera_sensitivity = DEFAULT_PREFS["camera_sensitivity"]
            prefs.cell_complex_size_mode = str(
                data.get("cell_complex_size_mode", prefs.cell_complex_size_mode)
            )
            if prefs.cell_complex_size_mode not in {"screen", "world"}:
                prefs.cell_complex_size_mode = DEFAULT_PREFS["cell_complex_size_mode"]
            for attr in (
                "cell_complex_screen_size_px",
                "cell_complex_world_size",
                "cell_complex_edge_width",
                "cell_complex_reference_label_text_size_px",
            ):
                try:
                    setattr(prefs, attr, float(data.get(attr, getattr(prefs, attr))))
                except (TypeError, ValueError):
                    setattr(prefs, attr, DEFAULT_PREFS[attr])
            for attr in ("cell_complex_default_color", "selection_color", "viewport_text_color"):
                color = data.get(attr, getattr(prefs, attr))
                if isinstance(color, list) and len(color) == 3:
                    try:
                        setattr(prefs, attr, [
                            int(max(0, min(255, int(c)))) for c in color
                        ])
                    except (TypeError, ValueError):
                        setattr(prefs, attr, list(DEFAULT_PREFS[attr]))
            prefs.region_growing_settings = _tool_settings_from_data(data, REGION_GROWING_PREFIX)
            prefs.plane_ransac_settings = _tool_settings_from_data(data, PLANE_RANSAC_PREFIX)
            prefs.superpoints_settings = _tool_settings_from_data(data, SUPERPOINTS_PREFIX)
            prefs.region_merge_settings = _tool_settings_from_data(data, REGION_MERGE_PREFIX)
        return prefs

    def set_region_growing_settings(self, settings: Dict[str, Any]) -> None:
        self.region_growing_settings = _sanitize_settings(settings)

    def get_region_growing_settings(self) -> Dict[str, Any]:
        return dict(self.region_growing_settings)

    def set_plane_ransac_settings(self, settings: Dict[str, Any]) -> None:
        self.plane_ransac_settings = _sanitize_settings(settings)

    def get_plane_ransac_settings(self) -> Dict[str, Any]:
        return dict(self.plane_ransac_settings)

    def set_superpoints_settings(self, settings: Dict[str, Any]) -> None:
        self.superpoints_settings = _sanitize_settings(settings)

    def get_superpoints_settings(self) -> Dict[str, Any]:
        return dict(self.superpoints_settings)

    def set_region_merge_settings(self, settings: Dict[str, Any]) -> None:
        self.region_merge_settings = _sanitize_settings(settings)

    def get_region_merge_settings(self) -> Dict[str, Any]:
        return dict(self.region_merge_settings)

    def to_toml(self) -> str:
        lines = [
            f'user_name = {_toml_scalar(self.user_name)}',
            f'enable_telemetry = {_toml_scalar(self.enable_telemetry)}',
            f'camera_sensitivity = {_toml_scalar(float(self.camera_sensitivity))}',
            f'cell_complex_size_mode = {_toml_scalar(self.cell_complex_size_mode)}',
            f'cell_complex_screen_size_px = {_toml_scalar(float(self.cell_complex_screen_size_px))}',
            f'cell_complex_world_size = {_toml_scalar(float(self.cell_complex_world_size))}',
            f'cell_complex_edge_width = {_toml_scalar(float(self.cell_complex_edge_width))}',
            f'cell_complex_reference_label_text_size_px = {_toml_scalar(float(self.cell_complex_reference_label_text_size_px))}',
            f'cell_complex_default_color = {_toml_scalar(self.cell_complex_default_color)}',
            f'selection_color = {_toml_scalar(self.selection_color)}',
            f'viewport_text_color = {_toml_scalar(self.viewport_text_color)}',
        ]
        tool_settings: Iterable[tuple[str, Dict[str, Any]]] = (
            (REGION_GROWING_PREFIX, self.region_growing_settings),
            (PLANE_RANSAC_PREFIX, self.plane_ransac_settings),
            (SUPERPOINTS_PREFIX, self.superpoints_settings),
            (REGION_MERGE_PREFIX, self.region_merge_settings),
        )
        for prefix, settings in tool_settings:
            for key in sorted(settings):
                lines.append(f"{prefix}{key} = {_toml_scalar(settings[key])}")
        return "\n".join(lines) + "\n"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_toml(), encoding="utf-8")
