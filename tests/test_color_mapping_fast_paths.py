from copy import deepcopy

import numpy as np
import pytest
import vtk
from PyQt6.QtWidgets import QApplication
from vtk.util import numpy_support as ns

from geon.data.definitions import ColorMap, vector_preset
from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering import color_mapping


@pytest.fixture(scope='session')
def app():
    return QApplication.instance() or QApplication([])


def test_exact_linear_detection():
    mapping = vector_preset(0, 255)
    assert color_mapping.linear_endpoints(mapping) == ((0.,)*3, (1.,)*3)
    mapping.red.color_positions = np.array([0., .25, 1.])
    mapping.red.output_values = np.array([0., .25, 1.])
    assert color_mapping.linear_endpoints(mapping) is not None
    mapping.red.output_values[1] += 1e-10
    assert color_mapping.linear_endpoints(mapping) is None
    gray = ColorMap.get_cmap('gray')
    gray.color_definitions = np.array([[1., 1., 1.], [0., 0., 0.]])
    assert color_mapping.linear_endpoints(gray) == ((1.,)*3, (0.,)*3)
    gray.color_type = 'hsv'
    assert color_mapping.linear_endpoints(gray) is None
    assert color_mapping.linear_endpoints(ColorMap.get_cmap('rainbow')) is None


@pytest.mark.parametrize('rgb,intensity,limits', [
    ([20., 120., 240.], .3, (0., 1.)),
    ([-30., 300., 80.], 2., (0., 1.)),
    ([20., 120., 240.], 4., (4., 4.)),
    ([np.nan, 120., 240.], np.inf, (0., 1.)),
])
def test_gpu_fast_and_lookup_blending_match(app, monkeypatch, rgb, intensity, limits):
    data = PointCloudData(np.zeros((1, 3), dtype=np.float32))
    data.add_field('rgb', np.array([rgb]), FieldType.COLOR)
    data.add_field('intensity', np.array([[intensity]]), FieldType.INTENSITY)
    data.get_fields('rgb')[0].color_map = vector_preset(0, 255)
    gray = ColorMap.get_cmap('gray')
    gray.min, gray.max = limits
    data.get_fields('intensity')[0].color_map = gray
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(60, 60)
    window.AddRenderer(renderer)
    layer.attach(renderer)
    layer.set_active_field_name('rgb')
    layer.set_background_field_name('intensity')
    layer.set_foreground_mix(.35)
    layer.set_point_size(15)
    renderer.ResetCamera()

    def pixel():
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        return ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(60, 60, -1)[30, 30, :3].copy()

    def buffer_times():
        return [(poly.GetPoints().GetMTime(),
                 poly.GetPointData().GetArray('geonForegroundInput').GetMTime(),
                 poly.GetPointData().GetArray('geonBackgroundInput').GetMTime())
                for poly, _, _ in layer._blend.pipelines]

    try:
        fast = pixel()
        before = buffer_times()
        mapper = layer._color_mapping
        detect = color_mapping.linear_endpoints
        monkeypatch.setattr(color_mapping, 'linear_endpoints', lambda _: None)
        for slot, (field, _) in mapper.states.items():
            mapper.update_map(slot, field.color_map)
        np.testing.assert_allclose(pixel(), fast, atol=1)
        monkeypatch.setattr(color_mapping, 'linear_endpoints', detect)
        # Preview a nonlinear curve, then restore the linear map on the same buffers.
        draft = deepcopy(data.get_fields('rgb')[0].color_map)
        draft.red.color_positions = np.array([0., .5, 1.])
        draft.red.output_values = np.array([0., .9, 1.])
        mapper.update_map('Foreground', draft)
        pixel()
        mapper.update_map('Foreground', data.get_fields('rgb')[0].color_map)
        mapper.update_map('Background', gray)
        np.testing.assert_allclose(pixel(), fast, atol=1)
        assert buffer_times() == before
    finally:
        layer.detach()
        window.Finalize()
