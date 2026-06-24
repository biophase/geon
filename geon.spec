# -*- mode: python ; coding: utf-8 -*-
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('geon')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

geon_spec = find_spec('geon')
if geon_spec is None or geon_spec.submodule_search_locations is None:
    raise RuntimeError("Could not locate installed geon package for PyInstaller build.")
geon_pkg_dir = Path(next(iter(geon_spec.submodule_search_locations))).resolve()
app_path = geon_pkg_dir / 'app.py'
icon_path = geon_pkg_dir / 'resources' / 'geon_icon.png'
if not icon_path.exists():
    icon_path = Path.cwd() / 'resources' / 'geon_icon.png'
icon = [str(icon_path)] if icon_path.exists() else None


a = Analysis(
    [str(app_path)],
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
    name='geon',
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
    icon=icon,
)
