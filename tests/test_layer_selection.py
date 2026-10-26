import os
from types import SimpleNamespace

import numpy as np
import pytest
import vtk

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication
from geon.data.pointcloud import PointCloudData, FieldType
from geon.data.cellcomplex import CellComplexData, VertexCell, EdgeCell
from geon.data.boundingbox import BoundingBoxData, BoundingBox
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.rendering.mesh import MeshLayer
from geon.rendering.selection import SelectableLayer
from geon.rendering.scene import Scene


@pytest.fixture(scope="module", autouse=True)
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def points():
    data = PointCloudData(np.arange(12, dtype=np.float32).reshape(4, 3))
    data.add_field("values", np.arange(4, dtype=np.float32)[:, None], FieldType.SCALAR)
    layer = PointCloudLayer(data)
    layer.active_selection = np.array([3, 1], dtype=np.int32)
    return layer


def test_point_extraction_is_independent_and_complete():
    source = points()
    extracted = source.extract_selection()
    assert type(extracted) is type(source)
    assert extracted.renderer is None and not extracted.has_selection
    assert extracted.id != source.id
    np.testing.assert_array_equal(extracted.data.points, source.data.points[[3, 1]])
    np.testing.assert_array_equal(extracted.data["values"], [[3], [1]])
    extracted.data.points[:] = 0
    extracted.data["values"][:] = 20
    assert source.data.points[3, 0] == 9
    assert source.data["values"][3, 0] == 3
    source.clear_selection()
    assert not source.has_selection and source.active_selection is None
    assert not source._selection_shader.selection_timer.isActive()
    with pytest.raises(ValueError, match="no selected"):
        source.extract_selection()


def test_edge_extraction_includes_boundaries_and_copies_attributes():
    vertices = [VertexCell((i, 0, 0)) for i in range(3)]
    edge = EdgeCell([vertices[0].id, vertices[1].id], attributes={"nested": [1]})
    layer = CellComplexLayer(CellComplexData(vertices, [edge]))
    layer.active_selection = {edge.id}
    copied = layer.extract_selection()
    assert len(copied.data.vertices) == 2 and len(copied.data.edges) == 1
    copied.data.normalize_and_validate()
    copied.data.edges[0].attributes["nested"].append(2)
    assert edge.attributes["nested"] == [1]
    assert layer.active_selection == {edge.id}
    layer._gizmo_hover_handle = "x"
    layer.clear_selection()
    assert not layer.has_selection and layer._gizmo_hover_handle is None
    assert layer.node_gizmo_origin is None
    assert not layer._selection_timer.isActive()


def test_box_face_clear_and_snapshot():
    box = BoundingBox(attributes={"nested": [1]})
    layer = BoundingBoxLayer(BoundingBoxData([box, BoundingBox()]))
    layer.selected_face = (box.id, "top")
    state = layer.capture_selection()
    copied = layer.extract_selection()
    assert len(copied.data.boxes) == 1
    assert copied.selected_face is None and not copied.has_selection
    copied.data.boxes[0].attributes["nested"].append(2)
    assert box.attributes["nested"] == [1]
    layer.clear_selection()
    assert layer.selected_face is None and not layer.has_selection
    assert not layer._selection_timer.isActive()
    layer.restore_selection(state)
    assert layer.selected_face == (box.id, "top")
    assert layer.has_selection


def test_capability_is_optional():
    assert not issubclass(MeshLayer, SelectableLayer)


def test_deselect_command_restores_box_face():
    from geon.tools.selection import ClearSelectionCmd
    from geon.tools.command_manager import CommandManager
    box = BoundingBox()
    layer = BoundingBoxLayer(BoundingBoxData([box]))
    layer.selected_face = (box.id, "top")
    class Context:
        controller = SimpleNamespace(layer_internal_sel_changed=SimpleNamespace(emit=lambda *_: None))
        viewer = SimpleNamespace(rerender=lambda: None)
    ctx = Context()
    manager = CommandManager()
    manager.do(ClearSelectionCmd(layer, ctx))
    assert not layer.has_selection and layer.selected_face is None
    manager.undo()
    assert layer.selected_face == (box.id, "top")
    manager.redo()
    assert not layer.has_selection and layer.selected_face is None


@pytest.mark.parametrize("choice, expected", [("active", 1), ("all", 2), ("cancel", 0)])
def test_extract_menu_multiple_layers_undo_redo(monkeypatch, choice, expected):
    from geon.ui import main_window
    from geon.tools.command_manager import CommandManager
    scene = Scene(vtk.vtkRenderer())
    sources = [points(), points()]
    for layer in sources:
        scene.doc.add_data(layer.data)
        scene.add_layer(layer)
    scene.active_layer_id = sources[0].id
    command_manager = CommandManager()
    emit = SimpleNamespace(emit=lambda *_: None)
    harness = SimpleNamespace(
        scene_manager=SimpleNamespace(_scene=scene, populate_tree=lambda: None,
                                      broadcastActivatedLayer=emit),
        dataset_manager=SimpleNamespace(populate_tree=lambda: None),
        viewer=SimpleNamespace(rerender=lambda: None),
        tool_controller=SimpleNamespace(do_global=command_manager.do),
        _mark_active_doc_modified=lambda: None,
    )
    real_box = main_window.QMessageBox

    class FakeBox:
        ButtonRole = real_box.ButtonRole
        StandardButton = real_box.StandardButton
        def __init__(self, parent):
            self.buttons = []
        def setWindowTitle(self, text): pass
        def setText(self, text): pass
        def exec(self): pass
        def addButton(self, *args):
            button = SimpleNamespace(setEnabled=lambda value: None)
            self.buttons.append(button)
            return button
        def clickedButton(self):
            return self.buttons[{"active": 0, "all": 1, "cancel": 2}[choice]]

    monkeypatch.setattr(main_window, "QMessageBox", FakeBox)
    try:
        main_window.MainWindow._on_extract_selection(harness)
        assert len(scene.layers) == 2 + expected
        assert len(scene.doc.scene_items) == 2 + expected
        assert all(source.has_selection for source in sources)
        if expected:
            new_ids = set(scene.layers) - {source.id for source in sources}
            command_manager.undo()
            assert len(scene.layers) == 2
            assert scene.active_layer is sources[0]
            command_manager.redo()
            assert new_ids <= set(scene.layers)
            assert all(not scene.layers[key].has_selection for key in new_ids)
    finally:
        for layer in list(scene.layers.values()):
            scene.remove_layer(layer.id)
