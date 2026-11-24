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


import subprocess
import pytest


@pytest.fixture(scope='session')
def vtk_rendering_available():
    """Probe Windows OpenGL in isolation: unsupported drivers may crash VTK."""
    if sys.platform != 'win32':
        return
    code = '''
import vtk
w = vtk.vtkRenderWindow()
w.SetOffScreenRendering(1)
w.SetSize(32, 32)
r = vtk.vtkRenderer()
w.AddRenderer(r)
if not w.SupportsOpenGL():
    print(w.GetOpenGLSupportMessage())
    raise SystemExit(77)
w.Render()
w.Finalize()
'''
    try:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.skip('Windows offscreen OpenGL capability probe timed out')
    if result.returncode:
        pytest.skip(f'Windows offscreen OpenGL unavailable (probe exit {result.returncode}): '
                    + (result.stdout + result.stderr)[-1200:])
