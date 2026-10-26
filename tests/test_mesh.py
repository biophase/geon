import numpy as np
import pytest
import vtk
from plyfile import PlyData, PlyElement

from geon.data.mesh import MeshData
from geon.data.document import Document
from geon.rendering.mesh import MeshLayer
from geon.rendering.layer_registry import LAYER_REGISTRY


VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)


@pytest.mark.parametrize("text", [True, False])
@pytest.mark.parametrize("index_name", ["vertex_indices", "vertex_index"])
def test_ply_polygons_and_document_round_trip(tmp_path, text, index_name):
    vertices = np.array([tuple(v) for v in VERTICES],
                        dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    faces = np.empty(2, dtype=[(index_name, object)])
    faces[index_name] = [np.array([0, 1, 2, 3]), np.array([0, 2, 3])]
    path = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(vertices, "vertex"),
             PlyElement.describe(faces, "face")], text=text).write(str(path))
    mesh = MeshData.from_ply(path)
    assert [len(f) for f in mesh.faces] == [4, 3]
    doc = Document("test")
    doc.add_data(mesh)
    target = tmp_path / "mesh.h5"
    doc.save_hdf5(target)
    loaded = Document.load_hdf5(target).scene_items[mesh.id]
    assert isinstance(loaded, MeshData)
    assert loaded.name == "mesh"
    np.testing.assert_array_equal(loaded.vertices, VERTICES)
    for actual, expected in zip(loaded.faces, mesh.faces):
        np.testing.assert_array_equal(actual, expected)
    assert loaded.get_extents() == (0, 1, 0, 1, 0, 0)


@pytest.mark.parametrize("faces", [[[0, 1]], [[0, 1, 4]], [[-1, 1, 2]],
                                   [[0., 1., 2.]], [[0, 1, 1]]])
def test_invalid_faces(faces):
    with pytest.raises(ValueError):
        MeshData(VERTICES, faces)


def test_quad_layer_renders_on_attach():
    mesh = MeshData(VERTICES, [[0, 1, 2, 3]], "Quad")
    layer = LAYER_REGISTRY.create_layer_for(mesh)
    assert isinstance(layer, MeshLayer)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(80, 80)
    window.AddRenderer(renderer)
    errors = []
    try:
        layer.attach(renderer)
        layer._mapper.AddObserver("ErrorEvent", lambda *_: errors.append(True))
        renderer.ResetCamera()
        window.Render()
        assert not errors
        assert layer._poly.GetNumberOfPolys() == 1
        assert layer._poly.GetCell(0).GetNumberOfPoints() == 4
        assert layer.world_xyz_from_pick(0, layer._actor, "cell") == (.5, .5, 0)
        assert layer.world_xyz_from_picked_id(2) == (1, 1, 0)
        layer.set_visible(False)
        assert not layer._actor.GetVisibility()
        layer.detach()
        assert renderer.GetActors().GetNumberOfItems() == 0
    finally:
        layer.detach()
        window.Finalize()


def test_empty_mesh():
    mesh = MeshData([], [])
    assert mesh.get_extents() is None


def test_point_only_ply_is_rejected(tmp_path):
    path = tmp_path / "points.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "end_header\n0 0 0\n")
    with pytest.raises(ValueError, match="vertex and face"):
        MeshData.from_ply(path)


def test_mesh_scene_lifecycle():
    from geon.rendering.scene import Scene
    mesh = MeshData(VERTICES, [[0, 1, 2, 3]])
    doc = Document("mesh")
    doc.add_data(mesh)
    scene = Scene(vtk.vtkRenderer())
    scene.set_document(doc)
    layer = scene.layers[mesh.id]
    assert scene.layer_for_prop(layer._actor) is layer
    scene.remove_layer(mesh.id)
    assert not doc.scene_items


def test_import_dialog():
    from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox
    from geon.ui.mesh_import_dialog import MeshImportDialog
    app = QApplication.instance() or QApplication([])
    mesh = MeshData(VERTICES, [[0, 1, 2, 3]])
    dialog = MeshImportDialog(mesh)
    assert not dialog.append_to_active_document
    assert dialog.destination_combo.count() == 1
    dialog.name_edit.setText(" ")
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.name_edit.setText("Building")
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert mesh.name == "Building"
    dialog.close()


def test_import_dialog_destination():
    from PyQt6.QtWidgets import QApplication
    from geon.ui.mesh_import_dialog import MeshImportDialog
    app = QApplication.instance() or QApplication([])
    mesh = MeshData(VERTICES, [[0, 1, 2, 3]])
    dialog = MeshImportDialog(mesh, active_document_name="Existing building")
    assert dialog.append_to_active_document
    assert "Existing building" in dialog.destination_combo.currentText()
    dialog.destination_combo.setCurrentIndex(0)
    assert not dialog.append_to_active_document
    dialog.reject()
    assert mesh.name == "Mesh"
    dialog.close()
