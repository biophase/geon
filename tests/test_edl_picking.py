from types import SimpleNamespace

import numpy as np
import pytest
import vtk

from geon.data.pointcloud import PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.picker import PointPicker
from geon.ui.viewer import VTKViewer


@pytest.mark.parametrize('association', ['point', 'cell'])
def test_edl_toggle_keeps_picking_and_releases_resources(association, capfd):
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(100, 100)
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    layer = PointCloudLayer(PointCloudData(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32)))
    layer.attach(renderer)
    layer.update()
    renderer.ResetCamera()
    picker = PointPicker(renderer)
    pick = picker.pick_point if association == 'point' else picker.pick_cell
    # Exercise the actual viewer toggle methods with an offscreen render window.
    viewer = SimpleNamespace(_renderer=renderer, _edl_pass=None,
                             vtkWidget=SimpleNamespace(GetRenderWindow=lambda: window),
                             rerender=window.Render)
    try:
        window.Render()
        assert pick(interactor, 50, 50).element_id == 0
        for _ in range(3):
            VTKViewer.enable_edl(viewer)
            edl = viewer._edl_pass
            VTKViewer.enable_edl(viewer)  # Idempotent; don't leak the previous pass.
            assert viewer._edl_pass is edl
            assert pick(interactor, 50, 50).element_id == 0
            assert renderer.GetPass() is edl
            layer.set_visibility_mask(np.array([False, True]))
            window.Render()
            result = pick(interactor, 50, 50)
            assert result.prop is layer._main_actor
            assert result.element_id == 1
            layer.set_visibility_mask(np.array([False, False]))
            assert pick(interactor, 50, 50) is None
            layer.set_visibility_mask(None)
            VTKViewer.disable_edl(viewer)
            assert renderer.GetPass() is None
            assert viewer._edl_pass is None
            del edl
            assert pick(interactor, 50, 50).element_id == 0
        assert 'should have been deleted' not in capfd.readouterr().err
    finally:
        if viewer._edl_pass is not None:
            VTKViewer.disable_edl(viewer)
        layer.detach()
        window.Finalize()


def test_pick_restores_display_pass_on_failure():
    renderer = vtk.vtkRenderer()
    display_pass = vtk.vtkRenderStepsPass()
    renderer.SetPass(display_pass)
    picker = PointPicker(renderer)
    interactor = SimpleNamespace(GetRenderWindow=lambda: SimpleNamespace(GetSize=lambda: (100, 100)))

    def fail():
        assert renderer.GetPass() is None
        raise RuntimeError('selection failed')

    selector = SimpleNamespace(SetArea=lambda *args: None, Select=fail)
    with pytest.raises(RuntimeError, match='selection failed'):
        picker._pick_with_selector(interactor, 50, 50, selector, 'point')
    assert renderer.GetPass() is display_pass
