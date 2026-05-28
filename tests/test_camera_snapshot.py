import h5py
import pytest
import vtk

from geon.data.camera import CameraData
from geon.data.document import Document
import geon.rendering.camera
from geon.rendering.camera import CameraLayer
from geon.rendering.layer_registry import LAYER_REGISTRY


def _configured_camera() -> vtk.vtkCamera:
    camera = vtk.vtkCamera()
    camera.SetPosition(1.0, 2.0, 3.0)
    camera.SetFocalPoint(4.0, 5.0, 6.0)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetClippingRange(0.2, 500.0)
    camera.SetViewAngle(42.0)
    camera.SetParallelProjection(True)
    camera.SetParallelScale(12.5)
    camera.SetWindowCenter(0.25, -0.5)
    camera.SetViewShear(0.1, 0.2, 0.9)
    return camera


def _assert_camera_matches(data: CameraData, camera: vtk.vtkCamera) -> None:
    assert camera.GetPosition() == pytest.approx(data.position)
    assert camera.GetFocalPoint() == pytest.approx(data.focal_point)
    assert camera.GetViewUp() == pytest.approx(data.view_up)
    assert camera.GetClippingRange() == pytest.approx(data.clipping_range)
    assert camera.GetViewAngle() == pytest.approx(data.view_angle)
    assert bool(camera.GetParallelProjection()) is data.parallel_projection
    assert camera.GetParallelScale() == pytest.approx(data.parallel_scale)
    assert camera.GetWindowCenter() == pytest.approx(data.window_center)
    assert camera.GetViewShear() == pytest.approx(data.view_shear)


def test_camera_data_hdf5_roundtrip_and_apply(tmp_path) -> None:
    source = _configured_camera()
    data = CameraData.from_camera(source, name="View A")

    path = tmp_path / "camera.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("camera")
        data.save_hdf5(group)

    with h5py.File(path, "r") as h5:
        loaded = CameraData.load_hdf5(h5["camera"])

    assert loaded.id == data.id
    assert loaded.name == "View A"
    assert loaded.position == pytest.approx(data.position)
    assert loaded.focal_point == pytest.approx(data.focal_point)
    assert loaded.view_up == pytest.approx(data.view_up)
    assert loaded.clipping_range == pytest.approx(data.clipping_range)
    assert loaded.view_angle == pytest.approx(data.view_angle)
    assert loaded.parallel_projection is True
    assert loaded.parallel_scale == pytest.approx(data.parallel_scale)
    assert loaded.window_center == pytest.approx(data.window_center)
    assert loaded.view_shear == pytest.approx(data.view_shear)

    target = vtk.vtkCamera()
    loaded.apply_to_camera(target)
    _assert_camera_matches(loaded, target)


def test_camera_data_json_roundtrip(tmp_path) -> None:
    data = CameraData.from_camera(_configured_camera(), name="JSON View")
    path = tmp_path / "camera.json"

    data.save_json(path)
    loaded = CameraData.load_json(path)

    assert loaded.id != data.id
    assert loaded.name == "JSON View"
    assert loaded.position == pytest.approx(data.position)
    assert loaded.focal_point == pytest.approx(data.focal_point)
    assert loaded.view_up == pytest.approx(data.view_up)
    assert loaded.clipping_range == pytest.approx(data.clipping_range)
    assert loaded.view_angle == pytest.approx(data.view_angle)
    assert loaded.parallel_projection is True
    assert loaded.parallel_scale == pytest.approx(data.parallel_scale)
    assert loaded.window_center == pytest.approx(data.window_center)
    assert loaded.view_shear == pytest.approx(data.view_shear)


def test_document_roundtrip_loads_camera_data(tmp_path) -> None:
    doc = Document("camera_doc")
    data = CameraData.from_camera(_configured_camera(), name="Stored View")
    doc.add_data(data)

    path = tmp_path / "doc.h5"
    doc.save_hdf5(path)

    loaded_doc = Document.load_hdf5(path)
    loaded = loaded_doc.scene_items[data.id]

    assert isinstance(loaded, CameraData)
    assert loaded.name == "Stored View"
    assert loaded.position == pytest.approx(data.position)


def test_camera_layer_registry_and_attach() -> None:
    data = CameraData.from_camera(_configured_camera())
    layer = LAYER_REGISTRY.create_layer_for(data)

    assert isinstance(layer, CameraLayer)

    renderer = vtk.vtkRenderer()
    layer.attach(renderer)

    assert list(layer.actors) == []
    assert layer.renderer is renderer
