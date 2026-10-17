from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as ns

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.layers.pointcloud_ui import _increase_point_size, _decrease_point_size


@pytest.mark.parametrize('coarse', [False, True])
def test_ribbon_size_preserves_buffers_and_changes_pixels(coarse):
    data = PointCloudData(np.zeros((1, 3), dtype=np.float32))
    data.add_field('color', np.array([[255, 0, 0]], dtype=np.uint8), FieldType.COLOR)
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(80, 80)
    window.AddRenderer(renderer)
    layer.attach(renderer)
    layer.update()
    layer._lod.close()
    layer._main_actor.SetMapper(layer._mapper_coarse if coarse else layer._mapper_fine)
    renderer.ResetCamera()
    label = SimpleNamespace(setText=Mock())
    render = Mock(side_effect=window.Render)
    controller = SimpleNamespace(ctx=SimpleNamespace(viewer=SimpleNamespace(rerender=render)))

    def pixels():
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.ShouldRerenderOff()
        capture.Update()
        return np.count_nonzero(ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())[:, 0])

    def data_times():
        return [(poly.GetMTime(), poly.GetPoints().GetMTime(), poly.GetVerts().GetMTime(),
                 poly.GetPointData().GetMTime()) for poly, _, _ in layer._blend.pipelines]

    try:
        window.Render()
        before = data_times()
        small = pixels()
        layer.update = Mock(side_effect=AssertionError('point size must not rebuild data'))
        _increase_point_size(layer, controller, label)
        assert layer.point_size == 3
        assert layer._main_actor.GetProperty().GetPointSize() == 3
        label.setText.assert_called_with('3')
        assert render.call_count == 1
        assert pixels() > small
        assert data_times() == before
        _decrease_point_size(layer, controller, label)
        assert pixels() == small
        assert render.call_count == 2
        layer.set_point_size(100)
        assert layer.point_size == 50
        property_time = layer._main_actor.GetProperty().GetMTime()
        _increase_point_size(layer, controller, label)
        assert layer._main_actor.GetProperty().GetMTime() == property_time
        assert render.call_count == 2
        layer.set_point_size(-1)
        assert layer.point_size == 1
        _decrease_point_size(layer, controller, label)
        label.setText.assert_called_with('1')
        assert render.call_count == 2
        assert data_times() == before
    finally:
        layer.detach()
        window.Finalize()
