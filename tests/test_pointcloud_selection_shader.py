import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication
from vtk.util import numpy_support as ns

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('color_mode', ['solid', 'rgb', 'scalar', 'blend'])
@pytest.mark.parametrize('coarse', [False, True])
@pytest.mark.parametrize('shader_visibility', [False, True])
def test_selection_color_visibility_and_picking(app, monkeypatch, color_mode, coarse, shader_visibility):
    monkeypatch.setenv('GEON_SHADER_VISIBILITY', '1' if shader_visibility else '0')
    points = np.full((201, 3), 100., dtype=np.float32)
    points[100] = [0, 0, 0]
    data = PointCloudData(points)
    if color_mode in ('rgb', 'blend'):
        data.add_field('color', np.tile([0, 0, 255], (201, 1)), FieldType.COLOR)
    elif color_mode == 'scalar':
        data.add_field('scalar', np.arange(201, dtype=np.float32).reshape(-1, 1), FieldType.SCALAR)
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(80, 80)
    window.AddRenderer(renderer)
    layer.attach(renderer)
    layer.update()
    if color_mode == 'blend':
        data.add_field('red', np.tile([255, 0, 0], (201, 1)), FieldType.COLOR)
        layer.set_background_field_name('red')
        layer.set_foreground_mix(.5)
    layer._lod.close()
    layer._main_actor.SetMapper(layer._mapper_coarse if coarse else layer._mapper_fine)
    layer._main_actor.GetProperty().SetPointSize(15)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 5)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2)
    camera.SetClippingRange(.1, 10)

    def pixel():
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        return ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(80, 80, -1)[40, 40, :3].copy()

    try:
        normal = pixel()
        layer.active_selection = np.array([100], dtype=np.int64)
        assert layer._selection_shader.selection_timer.isActive()
        assert renderer.GetActors().GetNumberOfItems() == 1
        before = [(poly.GetMTime(), poly.GetPoints().GetMTime(), poly.GetVerts().GetMTime())
                  for poly, _, _ in layer._blend.pipelines]
        layer._selection_shader.set_pulse(1)
        np.testing.assert_allclose(pixel(), [255, 128, 0], atol=2)
        layer._selection_shader.set_pulse(0)
        np.testing.assert_array_equal(pixel(), normal)
        layer._selection_shader.set_pulse(.5)
        np.testing.assert_allclose(pixel(), (normal.astype(float) + [255, 128, 0]) / 2, atol=2)
        assert before == [(poly.GetMTime(), poly.GetPoints().GetMTime(), poly.GetVerts().GetMTime())
                          for poly, _, _ in layer._blend.pipelines]
        selector = vtk.vtkOpenGLHardwareSelector()
        selector.SetRenderer(renderer)
        selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
        selector.SetArea(40, 40, 40, 40)
        result = selector.Select()
        assert result.GetNumberOfNodes() == 1
        picked = result.GetNode(0).GetSelectionList().GetValue(0)
        assert layer.data_index_from_picked_id(picked) == 100
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, .5)
        plane.SetNormal(0, 0, 1)
        layer.set_clipping_planes([plane])
        np.testing.assert_array_equal(pixel(), [0, 0, 0])
        layer.clear_clipping_planes()
        layer.set_visibility_mask(np.zeros(201, dtype=bool))
        np.testing.assert_array_equal(pixel(), [0, 0, 0])
        assert selector.Select().GetNumberOfNodes() == 0
        layer.set_visibility_mask(None)
        layer._selection_shader.set_pulse(1)
        np.testing.assert_allclose(pixel(), [255, 128, 0], atol=2)
        layer.active_selection = np.array([], dtype=np.int64)
        assert not layer._selection_shader.selection_timer.isActive()
        np.testing.assert_array_equal(pixel(), normal)
        layer.active_selection = np.array([0], dtype=np.int64)
        layer._selection_shader.set_pulse(1)
        np.testing.assert_array_equal(pixel(), normal)  # Unselected center unchanged.
        layer.active_selection = None
        assert not layer._selection_shader.selection_timer.isActive()
        # Reattachment reinstalls attributes and restarts a retained selection.
        layer.active_selection = np.array([100], dtype=np.int64)
        layer.detach()
        assert not layer._selection_shader.selection_timer.isActive()
        layer.attach(renderer)
        layer.update()
        layer._selection_shader.set_pulse(1)
        np.testing.assert_allclose(pixel(), [255, 128, 0], atol=2)
        assert renderer.GetActors().GetNumberOfItems() == 1
    finally:
        layer.detach()
        assert not layer._selection_shader.selection_timer.isActive()
        window.Finalize()
