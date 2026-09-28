import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as ns

from geon.data.pointcloud import PointCloudData, FieldType, SemanticSchema, SemanticClass
from geon.data.definitions import ColorMap
from geon.rendering.pointcloud import PointCloudLayer


def make_layer(count=201):
    data = PointCloudData(np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))).astype(np.float32))
    data.add_field("red", np.tile([255, 0, 0], (count, 1)), FieldType.COLOR)
    data.add_field("blue", np.tile([0, 0, 255], (count, 1)), FieldType.COLOR)
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    layer.attach(renderer)
    layer.update()
    return layer, renderer


def test_mix_only_changes_uniform_and_preserves_state():
    layer, _ = make_layer()
    layer.set_background_field_name("blue")
    shader = layer._main_actor.GetShaderProperty()
    def times():
        return [shader.GetShaderMTime(), *[
            obj.GetMTime() for poly, mapper, _ in layer._blend.pipelines
            for obj in (poly, poly.GetPoints(), poly.GetPointData().GetScalars(),
                        poly.GetPointData().GetArray(layer._blend.ARRAY), mapper)
        ]]
    before = times()
    for value in [0, 0.25, 0.5, 1, 2, -1]:
        layer.set_foreground_mix(value)
        assert times() == before
    assert layer.foreground_mix == 0
    layer.set_foreground_mix(.37)
    layer.set_active_field_name("blue")
    assert layer.background_field_name == "blue"
    layer.set_background_field_name(None)
    assert layer.foreground_mix == .37
    assert not layer._blend.enabled
    for poly, mapper, _ in layer._blend.pipelines:
        assert poly.GetPointData().GetArray(layer._blend.ARRAY) is None
    layer.set_background_field_name("red")
    layer.data.remove_fields("red")
    layer.update()
    assert layer.background_field_name is None


@pytest.mark.parametrize("kind", [FieldType.SCALAR, FieldType.INTENSITY, FieldType.VECTOR, FieldType.NORMAL, FieldType.INSTANCE])
def test_field_conversion_visibility_and_lod(kind):
    layer, _ = make_layer()
    values = np.arange(201, dtype=np.float32).reshape(-1, 1)
    if kind in (FieldType.VECTOR, FieldType.NORMAL):
        values = np.column_stack((values, values[::-1], values))
    if kind == FieldType.INSTANCE:
        values = values.astype(np.int32)
    layer.data.add_field("field", values, kind)
    layer.set_background_field_name("field")
    mask = np.arange(201) % 2 == 0
    layer.set_visibility_mask(mask)
    assert layer._poly.GetNumberOfPoints() == 101
    assert layer._poly_coarse.GetNumberOfPoints() == 2
    fine = ns.vtk_to_numpy(layer._poly.GetPointData().GetArray(layer._blend.ARRAY))
    coarse = ns.vtk_to_numpy(layer._poly_coarse.GetPointData().GetArray(layer._blend.ARRAY))
    np.testing.assert_equal(coarse, fine[::100])
    layer.set_visibility_mask(np.zeros(201, dtype=bool))
    assert layer._poly.GetNumberOfPoints() == 0
    layer.set_background_field_name(None)
    layer.set_active_field_name("field")


@pytest.mark.parametrize("kind", list(FieldType))
@pytest.mark.parametrize("coarse", [False, True])
def test_gpu_pixels_endpoints_midpoint_and_fallback(kind, coarse):
    layer, renderer = make_layer(1)
    background_name = "blue"
    if kind != FieldType.COLOR:
        values = np.array([[.25, .75, .5]]) if kind in (FieldType.VECTOR, FieldType.NORMAL) else np.array([[0]])
        schema = SemanticSchema("test", [SemanticClass(0, "class", (0, 255, 0))])
        layer.data.add_field("other", values, kind, schema=schema)
        layer.data.get_fields("other")[0].color_map = ColorMap("test")
        background_name = "other"
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(80, 80)
    window.AddRenderer(renderer)
    layer.set_point_size(20)
    renderer.ResetCamera()
    if coarse:
        renderer.AddObserver(vtk.vtkCommand.StartEvent,
                             lambda *_: renderer.SetAllocatedRenderTime(1e-12), 1.0)
    errors = []
    layer._mapper_fine.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))

    def pixel():
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        return ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(80, 80, -1)[40, 40, :3]

    try:
        red = pixel().copy()
        layer.set_active_field_name(background_name)
        blue = pixel().copy()
        layer.set_active_field_name("red")
        layer.set_background_field_name(background_name)
        for mix, expected in [(0, blue), (1, red), (.5, (red.astype(float) + blue) / 2)]:
            layer.set_foreground_mix(mix)
            np.testing.assert_allclose(pixel(), expected, atol=1)
        if coarse:
            assert layer._main_actor.GetMapper() is layer._mapper_coarse
        assert not errors
        layer.set_background_field_name(None)
        np.testing.assert_equal(pixel(), red)
    finally:
        window.Finalize()


def test_live_background_edits_components_and_colormap():
    layer, _ = make_layer(2)
    layer.data.add_field("vector", np.array([[0., 1., 0.], [1., 0., 0.]]), FieldType.VECTOR)
    layer.set_background_field_name("vector")
    def background():
        return ns.vtk_to_numpy(layer._poly.GetPointData().GetArray(layer._blend.ARRAY)).copy()
    original = background()
    layer.set_vector_field_active_index("vector", 1)
    np.testing.assert_allclose(background(), original[::-1])
    layer.data.get_fields("vector")[0].color_map = ColorMap("new")
    layer.update()
    assert not np.array_equal(background(), original[::-1])
    schema = SemanticSchema("test", [SemanticClass(0, "class", (0, 255, 0))])
    layer.data.add_field("semantic", np.zeros(2, dtype=np.int32), FieldType.SEMANTIC, schema=schema)
    layer.set_background_field_name("semantic")
    np.testing.assert_equal(background(), [[0, 1, 0], [0, 1, 0]])
    schema.semantic_classes[0].color = (255, 0, 0)
    layer.update()
    np.testing.assert_equal(background(), [[1, 0, 0], [1, 0, 0]])


def test_lod_picking_clipping_and_detach():
    layer, renderer = make_layer()
    layer.set_background_field_name("blue")
    mask = np.arange(201) % 2 == 0
    layer.set_visibility_mask(mask)
    plane = vtk.vtkPlane()
    layer.set_clipping_planes([plane])
    for _, mapper, _ in layer._blend.pipelines:
        assert mapper.GetNumberOfClippingPlanes() == 1
    layer._main_actor.SetMapper(layer._mapper_coarse)
    assert layer.data_index_from_picked_id(1) == 200
    assert layer.world_xyz_from_picked_id(1) == (200., 0., 0.)
    layer.detach()
    assert not renderer.HasObserver(vtk.vtkCommand.StartEvent)
    layer.attach(renderer)
    layer.update()
    assert layer._blend.enabled
