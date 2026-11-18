"""Run kernels in fresh processes to detect native symbol/import-order conflicts."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('ifc_first', [False, True])
def test_occt_ifcopenshell_coexistence(ifc_first):
    import geon._native
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'smoke_surface_geometry.py'
    # Preserve the package under test, including source builds with out-of-tree
    # native modules. Installed-wheel CI runs with GEON_USE_INSTALLED=1.
    code = (
        "import sys,runpy; "
        "sys.meta_path[:]=[f for f in sys.meta_path if f.__class__.__name__!='ScikitBuildRedirectingFinder']; "
        f"sys.path[:]={sys.path!r}; import geon._native; "
        f"geon._native.__path__[:]={list(geon._native.__path__)!r}; "
        f"sys.argv={[str(script)] + (['--ifc-first'] if ifc_first else [])!r}; "
        f"runpy.run_path({str(script)!r},run_name='__main__')"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'smoke test passed' in result.stdout
