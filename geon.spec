# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
import tempfile

from PyInstaller.utils.hooks import collect_all

import geon


datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all("geon")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]


GEON_ROOT = Path(geon.__file__).resolve().parent
APP_PATH = GEON_ROOT / "app.py"
ICON_SOURCE = GEON_ROOT / "resources" / "geon_icon.png"


def _resolve_icon() -> str | None:
    if not ICON_SOURCE.exists():
        return None
    if not sys.platform.startswith("win"):
        return str(ICON_SOURCE)

    try:
        from PIL import Image
    except Exception:
        return None

    icon_dir = Path(tempfile.gettempdir()) / "geon_pyinstaller"
    icon_dir.mkdir(parents=True, exist_ok=True)
    icon_path = icon_dir / "geon_icon.ico"
    with Image.open(ICON_SOURCE) as img:
        img.save(icon_path)
    return str(icon_path)


ICON_PATH = _resolve_icon()


a = Analysis(
    [str(APP_PATH)],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="geon",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH,
)
