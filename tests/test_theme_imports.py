import builtins
import importlib.util
import sys
from pathlib import Path

import pytest


def test_theme_import_is_safe_without_pyqt(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("PyQt6"):
            raise ImportError("PyQt6 intentionally blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    module_name = "_test_geon_config_theme"
    sys.modules.pop(module_name, None)
    theme_path = Path(__file__).resolve().parent.parent / "src" / "geon" / "config" / "theme.py"
    spec = importlib.util.spec_from_file_location(module_name, theme_path)
    assert spec is not None
    assert spec.loader is not None

    theme = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = theme
    spec.loader.exec_module(theme)

    assert theme.DEFAULT_OBJ_COLOR == (0.6, 0.6, 0.6)
    with pytest.raises(RuntimeError, match="set_dark_palette requires PyQt6"):
        theme.set_dark_palette(object())
