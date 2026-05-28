import vtk

from geon.data.camera import CameraData
from geon.rendering.scene import Scene
import geon.rendering.camera


def _camera_data(name: str) -> CameraData:
    return CameraData(name=name)


def test_remove_layer_deletes_document_data() -> None:
    scene = Scene(vtk.vtkRenderer())
    data = _camera_data("View A")
    scene.doc.add_data(data)
    layer = scene.add_data(data)

    scene.remove_layer(layer.id, delete_data=True)

    assert layer.id not in scene.layers
    assert data.id not in scene.doc.scene_items


def test_remove_active_layer_selects_next_layer() -> None:
    scene = Scene(vtk.vtkRenderer())
    first = _camera_data("View A")
    second = _camera_data("View B")
    scene.doc.add_data(first)
    scene.doc.add_data(second)
    first_layer = scene.add_data(first)
    second_layer = scene.add_data(second)
    scene.active_layer_id = first_layer.id

    scene.remove_layer(first_layer.id, delete_data=True)

    assert scene.active_layer is second_layer
    assert scene.active_layer_id == second_layer.id


def test_remove_last_active_layer_clears_active_layer() -> None:
    scene = Scene(vtk.vtkRenderer())
    data = _camera_data("View A")
    scene.doc.add_data(data)
    layer = scene.add_data(data)
    scene.active_layer_id = layer.id

    scene.remove_layer(layer.id, delete_data=True)

    assert scene.active_layer is None
    assert scene.active_layer_id is None
