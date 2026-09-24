from __future__ import annotations

import os
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QRadioButton, QTreeWidgetItem

from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.scene import Scene
from geon.settings import Preferences
from geon.tools import annotate
from geon.tools.annotate import AnnotatePointsCmd, AnnotateTool
from geon.tools.command_manager import CommandManager
from geon.tools.tool_context import ToolContext
from geon.ui.context_ribbon import ContextRibbon
from geon.ui.preferences_dialog import PreferencesDialog
from geon.ui.scene_manager import SceneManager


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def make_data():
    data = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    for name, kind in (("instances", FieldType.INSTANCE), ("other", FieldType.INSTANCE),
                       ("semantics", FieldType.SEMANTIC)):
        data.add_field(name=name, data=np.array([-1, 0, 2, 2], dtype=np.int32), field_type=kind)
    return data


@pytest.mark.parametrize("mode, expected", [("new", 1), ("unassigned", -1)])
@pytest.mark.parametrize("semantic", [False, True])
def test_annotation_undo_redo_keeps_selection_and_id(qt_app, mode, expected, semantic):
    data = make_data()
    layer = PointCloudLayer(data)
    layer.active_selection = np.array([1, 2], dtype=np.int32)
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    cmd = AnnotatePointsCmd(
        title="Annotate", sem_field_name="semantics" if semantic else None,
        inst_field_name="instances", sem_inds_old=None, inst_inds_old=None,
        sem_ind_new=4 if semantic else None, layer_ref=weakref.ref(layer),
        ctx_ref=weakref.ref(ctx), instance_mode=mode,
    )
    manager = CommandManager()
    manager.do(cmd)
    np.testing.assert_array_equal(data["instances"].ravel(), [-1, expected, expected, 2])
    np.testing.assert_array_equal(data["other"].ravel(), [-1, 0, 2, 2])
    np.testing.assert_array_equal(data["semantics"].ravel(), [-1, 4, 4, 2] if semantic else [-1, 0, 2, 2])
    manager.undo()
    np.testing.assert_array_equal(data["instances"].ravel(), [-1, 0, 2, 2])
    np.testing.assert_array_equal(data["semantics"].ravel(), [-1, 0, 2, 2])
    layer.active_selection = np.array([3], dtype=np.int32)
    data["instances"][0] = 1  # Change the next free ID before redo.
    manager.redo()
    np.testing.assert_array_equal(data["instances"].ravel(), [1, expected, expected, 2])
    manager.undo()
    np.testing.assert_array_equal(data["instances"].ravel(), [1, 0, 2, 2])


@pytest.mark.parametrize("selection", [None, np.array([], dtype=np.int32)])
def test_annotation_empty_selection_is_noop(qt_app, selection):
    layer = PointCloudLayer(make_data())
    layer._active_selection = selection
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    cmd = AnnotatePointsCmd("Annotate", None, "instances", None, None, None,
                            weakref.ref(layer), weakref.ref(ctx), "unassigned")
    cmd.execute()
    cmd.undo()
    np.testing.assert_array_equal(layer.data["instances"].ravel(), [-1, 0, 2, 2])


def test_ribbon_modes_and_height(qt_app, monkeypatch):
    monkeypatch.setattr(annotate, "_ANNOTATE_RIBBON_SESSION", annotate._AnnotateRibbonSessionState())
    layer = PointCloudLayer(make_data())
    layer.active_selection = np.array([0, 2], dtype=np.int32)
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    tool = AnnotateTool(CommandManager(), ctx)
    ribbon = ContextRibbon()
    content = tool.create_context_widget(ribbon)
    ribbon.set_group("Annotate", content, "tool")
    ribbon.show()
    qt_app.processEvents()
    radios = {radio.text(): radio for radio in content.findChildren(QRadioButton)}
    assert radios["new"].isChecked()
    assert not radios["unassigned"].isEnabled()
    layout = content.layout()
    layout.itemAtPosition(1, 0).widget().setChecked(True)
    layout.itemAtPosition(0, 0).widget().setChecked(False)
    radios["unassigned"].click()
    assert not radios["new"].isChecked()
    assert tool.choice_instance_mode == "unassigned"
    assert layout.rowCount() == 2
    assert radios["new"].y() == radios["unassigned"].y()
    assert ribbon.height() == ribbon.RIBBON_HEIGHT
    assert radios["unassigned"].geometry().bottom() < content.height()
    assert layout.itemAtPosition(0, 5).widget().text() == "Accept"
    assert layout.itemAtPosition(1, 5).widget().text() == "Copy from ..."
    restored = AnnotateTool(CommandManager(), ctx)
    assert restored.choice_instance_mode == "unassigned"
    layout.itemAtPosition(0, 5).widget().click()
    np.testing.assert_array_equal(layer.data["instances"].ravel(), [-1, 0, -1, 2])
    ctx.controller.deactivate_tool.assert_called_once()
    ribbon.clear_group("tool")
    assert ribbon.height() == ribbon.RIBBON_HEIGHT
    ribbon.close()


@pytest.mark.parametrize("field_name", ["instances", "semantics"])
def test_unassigned_rgba_visibility_and_opacity(qt_app, field_name):
    layer = PointCloudLayer(make_data())
    layer.active_field_name = field_name
    layer.attach(vtk.vtkRenderer())
    layer.update()
    colors = vtk_to_numpy(layer._poly.GetPointData().GetScalars()).copy()
    assert colors.shape == (4, 4)
    np.testing.assert_array_equal(colors[0], [204, 204, 204, 192])
    np.testing.assert_array_equal(colors[1:, 3], [255, 255, 255])
    layer.set_unassigned_point_color([10, 20, 30, 100])
    layer._visibility_mask = np.array([True, False, True, True])
    layer._temporary_point_opacity = np.array([0.5, 1, 0.5, 1], dtype=np.float32)
    layer.update()
    visible = vtk_to_numpy(layer._poly.GetPointData().GetScalars())
    assert visible.shape == (3, 4)
    np.testing.assert_array_equal(visible[0], [10, 20, 30, 50])
    np.testing.assert_array_equal(visible[1:, :3], colors[2:, :3])
    np.testing.assert_array_equal(visible[1:, 3], [128, 255])
    layer.detach()


def test_scene_preferences_apply_to_existing_and_new_layers(qt_app):
    scene = Scene(vtk.vtkRenderer(), unassigned_point_color=[1, 2, 3, 4])
    first = scene.add_data(make_data())
    first.update()
    np.testing.assert_array_equal(vtk_to_numpy(first._poly.GetPointData().GetScalars())[0], [1, 2, 3, 4])
    scene.set_unassigned_point_color([5, 6, 7, 8])
    second = scene.add_data(make_data())
    for layer in (first, second):
        layer.update()
        np.testing.assert_array_equal(vtk_to_numpy(layer._poly.GetPointData().GetScalars())[0], [5, 6, 7, 8])
    scene.clear(delete_data=False)


def test_preferences_sections_color_picker_and_cancel(qt_app, monkeypatch):
    prefs = Preferences()
    dialog = PreferencesDialog(prefs)
    assert [dialog.tabs.tabText(i) for i in range(3)] == ["General", "Viewport", "Appearance"]
    header, content = dialog.sections["Cell complex"]
    header.click()
    assert content.isHidden() and content.isEnabled()
    assert header.arrowType() == Qt.ArrowType.RightArrow
    header.click()
    assert not content.isHidden()
    picked = QColor(10, 20, 30, 40)
    monkeypatch.setattr("geon.ui.preferences_dialog.QColorDialog.getColor", lambda *args: picked)
    dialog.unassigned_color_button.click()
    assert dialog.unassigned_color_preview.text() == "10, 20, 30, 40"
    dialog.reject()
    assert prefs.unassigned_point_color == [204, 204, 204, 192]
    dialog.apply()
    assert prefs.unassigned_point_color == [10, 20, 30, 40]


def test_scene_tree_collapse_button(qt_app):
    manager = SceneManager(Mock(), Mock())
    root = QTreeWidgetItem(["Layer"])
    child = QTreeWidgetItem(root, ["Field"])
    QTreeWidgetItem(child, ["Class"])
    manager.tree.addTopLevelItem(root)
    manager.tree.expandAll()
    assert root.isExpanded() and child.isExpanded()
    assert not manager.collapse_button.icon().isNull()
    assert manager.collapse_button.height() == manager.collapse_button.fontMetrics().height()
    manager.collapse_button.click()
    assert not root.isExpanded() and not child.isExpanded()
    assert manager.tree.topLevelItemCount() == 1
    manager.close()


def test_rgb_view_keeps_colors_and_handles_empty_visibility(qt_app):
    data = make_data()
    rgb = np.array([[10, 20, 30]] * 4, dtype=np.uint8)
    data.add_field(name="rgb", data=rgb, field_type=FieldType.COLOR)
    layer = PointCloudLayer(data)
    layer.active_field_name = "rgb"
    layer.attach(vtk.vtkRenderer())
    layer.set_unassigned_point_color([1, 2, 3, 0])
    np.testing.assert_array_equal(vtk_to_numpy(layer._poly.GetPointData().GetScalars()), rgb)
    layer.active_field_name = "instances"
    layer._visibility_mask = np.zeros(4, dtype=bool)
    layer.update()
    assert layer._poly.GetNumberOfPoints() == 0
    assert layer._poly.GetPointData().GetScalars().GetNumberOfComponents() == 4
    layer.detach()


def test_scene_manager_applies_preferences_on_load_and_change(qt_app):
    from geon.data.document import Document

    viewer = Mock()
    viewer._renderer = vtk.vtkRenderer()
    manager = SceneManager(viewer, Mock())
    prefs = Preferences(unassigned_point_color=[10, 20, 30, 40])
    manager.preferences = prefs
    doc = Document()
    doc.add_data(make_data())
    manager.on_document_loaded(doc)
    layer = manager._scene.active_layer
    assert layer.unassigned_point_color == (10, 20, 30, 40)
    prefs.unassigned_point_color = [50, 60, 70, 80]
    manager.preferences = prefs
    np.testing.assert_array_equal(
        vtk_to_numpy(layer._poly.GetPointData().GetScalars())[0], [50, 60, 70, 80]
    )
    manager._scene.clear(delete_data=False)
    manager.close()
