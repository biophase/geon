from __future__ import annotations

import os
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import vtk
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget

from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.settings import Preferences
from geon.tools.annotate import AnnotatePointsCmd, AnnotateTool
from geon.tools.base import Event
from geon.tools.command_manager import CommandManager
from geon.tools.controller import ToolController
from geon.tools.rectangle import RectangleSelectionTool
from geon.tools.selection import SelectPointsCmd
from geon.tools.tool_context import ToolContext
from geon.ui.context_ribbon import ContextRibbon
from geon.ui.main_window import MainWindow
from geon.ui.menu_bar import MenuBar
from geon.ui.preferences_dialog import PreferencesDialog
from geon.ui.viewer import InteractorStyle, VTKViewer


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def make_layer(values=(-1, 0, 0, 2, 2, -1)):
    data = PointCloudData(np.zeros((len(values), 3), dtype=np.float32))
    data.add_field(name="instances", data=np.array(values, dtype=np.int32),
                   field_type=FieldType.INSTANCE)
    data.add_field(name="semantics", data=np.array(values, dtype=np.int32),
                   field_type=FieldType.SEMANTIC)
    layer = PointCloudLayer(data)
    layer.active_field_name = "instances"
    return layer


def test_split_preserves_independent_masks_and_redo(app):
    layer = make_layer()
    layer.active_selection = np.array([0, 1, 3])
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    cmd = AnnotatePointsCmd("Split", None, "instances", None, None, None,
                            weakref.ref(layer), weakref.ref(ctx), "split")
    manager = CommandManager()
    manager.do(cmd)
    values = layer.active_field.data.ravel().copy()
    assert values[0] == -1
    assert values[1] != values[3]
    assert values[1] not in {-1, 0, 2}
    assert values[3] not in {-1, 0, 2}
    np.testing.assert_array_equal(values[[2, 4, 5]], [0, 2, -1])
    manager.undo()
    np.testing.assert_array_equal(layer.active_field.data.ravel(), [-1, 0, 0, 2, 2, -1])
    layer.active_selection = np.array([5])
    manager.redo()
    np.testing.assert_array_equal(layer.active_field.data.ravel(), values)


@pytest.mark.parametrize("contract, expected", [(False, [1, 2, 3, 4]), (True, [1, 2])])
def test_instance_selection_and_history(app, contract, expected):
    layer = make_layer()
    layer.active_selection = np.array([1, 2, 3])
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    manager = CommandManager()
    result = layer.instance_selection(contract=contract)
    np.testing.assert_array_equal(result, expected)
    manager.do(SelectPointsCmd("Instances", result, weakref.ref(layer), weakref.ref(ctx)))
    manager.undo()
    np.testing.assert_array_equal(layer.active_selection, [1, 2, 3])
    manager.redo()
    np.testing.assert_array_equal(layer.active_selection, expected)


def test_instance_selection_empty_unassigned_and_wrong_field(app):
    layer = make_layer()
    assert layer.instance_selection().size == 0
    layer.active_selection = np.array([0])
    np.testing.assert_array_equal(layer.instance_selection(), [0, 5])
    assert layer.instance_selection(contract=True).size == 0
    layer.active_field_name = "semantics"
    with pytest.raises(ValueError, match="not of type instance"):
        layer.instance_selection()


def test_selection_redo_retains_original_none(app):
    layer = make_layer()
    ctx = ToolContext(SimpleNamespace(active_layer=layer), Mock(), Mock())
    manager = CommandManager()
    manager.do(SelectPointsCmd("Selection", np.array([1]), weakref.ref(layer), weakref.ref(ctx)))
    manager.undo()
    manager.redo()
    manager.undo()
    assert layer.active_selection is None


def test_instance_menu_and_error_popups(app, monkeypatch):
    menu = MenuBar(None)
    instance = next(action.menu() for action in menu.select_menu.actions() if action.text() == "Instance")
    assert [action.shortcut().toString() for action in instance.actions()] == ["Ctrl+E", "Ctrl+R"]
    warning = Mock()
    monkeypatch.setattr("geon.ui.main_window.QMessageBox.warning", warning)
    fake = SimpleNamespace(scene_manager=SimpleNamespace(_scene=SimpleNamespace(active_layer=object())))
    MainWindow._on_instance_selection(fake, False)
    assert "object does not implement instance selection contraction." in warning.call_args.args
    fake.scene_manager._scene.active_layer = make_layer()
    fake.scene_manager._scene.active_layer.active_field_name = "semantics"
    MainWindow._on_instance_selection(fake, True)
    assert "The active field is not of type instance." in warning.call_args.args


def rectangle_context(layer):
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetSize(100, 100)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 10)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(1)
    viewer = Mock(_renderer=renderer)
    controller = Mock()
    ctx = ToolContext(SimpleNamespace(active_layer=layer), viewer, controller)
    manager = CommandManager()
    controller.do_global.side_effect = manager.do
    tool = RectangleSelectionTool(manager, ctx)
    controller.deactivate_tool.side_effect = tool.deactivate
    return tool, ctx, manager, window


def click(x, y, key=None):
    return Event((x, y), (x, y), False, False, False, key)


def test_rectangle_preview_two_clicks_and_history(app):
    layer = make_layer()
    layer.data.points[:] = [[-.5, -.5, 0], [.5, .5, 0], [0, 0, 0],
                            [.9, .9, 0], [-.9, -.9, 0], [-.8, .8, 0]]
    tool, ctx, manager, window = rectangle_context(layer)
    tool._click_event(click(20, 20))
    tool._mouse_move_event(click(60, 60))
    assert tool.polygon_actor is not None
    assert tool.poly_points == [(20, 20)]
    tool._click_event(click(60, 60))
    np.testing.assert_array_equal(layer.active_selection, [0, 2])
    manager.undo()
    assert layer.active_selection is None
    manager.redo()
    np.testing.assert_array_equal(layer.active_selection, [0, 2])
    assert layer._visibility_mask is None
    window.Finalize()


def test_rectangle_escape_restores_selection_and_visibility(app):
    layer = make_layer()
    layer.active_selection = np.array([1, 2])
    mask = np.array([True, True, True, False, False, False])
    layer.set_visibility_mask(mask)
    tool, ctx, manager, window = rectangle_context(layer)
    tool._click_event(click(10, 10))
    tool._mouse_move_event(click(80, 80))
    tool.key_press_hook(click(80, 80, "Escape"))
    np.testing.assert_array_equal(layer.active_selection, [1, 2])
    np.testing.assert_array_equal(layer._visibility_mask, mask)
    assert not manager.undo_stack
    ctx.viewer.set_camera_enabled.assert_called_with(True)
    window.Finalize()


def test_annotation_accept_key_preserves_text_entry(app, monkeypatch):
    controller = ToolController(ContextRibbon())
    ctx = ToolContext(SimpleNamespace(active_layer=make_layer()), Mock(), controller)
    tool = AnnotateTool(CommandManager(), ctx)
    controller._active_tool = tool
    accept = Mock()
    monkeypatch.setattr(tool, "_Accept", accept)
    monkeypatch.setattr(QApplication, "focusWidget", lambda: None)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier)
    assert controller.eventFilter(None, event)
    accept.assert_called_once()
    editor = QLineEdit()
    monkeypatch.setattr(QApplication, "focusWidget", lambda: editor)
    assert not controller.eventFilter(None, event)
    accept.assert_called_once()


def test_precision_preferences_round_trip(app, tmp_path):
    prefs = Preferences(path=tmp_path / "prefs.toml")
    assert prefs.precision_camera_multiplier == .05
    assert prefs.precision_camera_hotkey == "Shift"
    dialog = PreferencesDialog(prefs)
    dialog.precision_camera_multiplier_input.setValue(.2)
    dialog.precision_camera_hotkey_input.setCurrentText("Alt")
    dialog.apply()
    prefs.save()
    loaded = Preferences.load(prefs.path)
    assert loaded.precision_camera_multiplier == .2
    assert loaded.precision_camera_hotkey == "Alt"
    dialog.close()


def test_camera_precision_scales_motion_and_restores(app):
    renderer = vtk.vtkRenderer()
    viewer = Mock()
    viewer.precision_camera_factor.return_value = .05
    style = InteractorStyle(renderer, viewer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 10)
    def move():
        x, y, z = camera.GetPosition()
        camera.SetPosition(x + 10, y, z)
        x, y, z = camera.GetFocalPoint()
        camera.SetFocalPoint(x + 10, y, z)
    style._camera_action(move)
    assert camera.GetPosition() == (.5, 0, 10)
    viewer.precision_camera_factor.return_value = 1.0
    style._camera_action(move)
    assert camera.GetPosition() == (10.5, 0, 10)
    assert style.GetMotionFactor() == 10


def test_precision_key_release_and_focus_loss(app, monkeypatch):
    # Initialize just the QWidget so this test needs no OpenGL context.
    viewer = VTKViewer.__new__(VTKViewer)
    QWidget.__init__(viewer)
    viewer.set_camera_precision(.05, "Shift")
    monkeypatch.setattr(QApplication, "queryKeyboardModifiers", lambda: Qt.KeyboardModifier.ShiftModifier)
    assert viewer.precision_camera_factor() == .05
    monkeypatch.setattr(QApplication, "queryKeyboardModifiers", lambda: Qt.KeyboardModifier.NoModifier)
    assert viewer.precision_camera_factor() == 1
    viewer.set_camera_precision(.2, "P")
    viewer.eventFilter(viewer, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
    assert viewer.precision_camera_factor() == .2
    viewer.eventFilter(viewer, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
    assert viewer.precision_camera_factor() == 1
    viewer.eventFilter(viewer, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.NoModifier))
    viewer.eventFilter(viewer, QEvent(QEvent.Type.FocusOut))
    assert viewer.precision_camera_factor() == 1
    viewer.close()


def test_precision_orbit_retains_distance_and_zoom_scales(app):
    renderer = vtk.vtkRenderer()
    viewer = Mock()
    viewer.precision_camera_factor.return_value = .05
    style = InteractorStyle(renderer, viewer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 10)
    style._camera_action(lambda: camera.Azimuth(20))
    assert camera.GetDistance() == pytest.approx(10)
    assert camera.GetPosition()[0] == pytest.approx(.1745, abs=.01)
    camera.SetParallelScale(10)
    style._camera_action(lambda: camera.SetParallelScale(20))
    assert camera.GetParallelScale() == pytest.approx(10.5)
    viewer.rerender.reset_mock()
    style._camera_action(lambda: None)
    viewer.rerender.assert_not_called()
