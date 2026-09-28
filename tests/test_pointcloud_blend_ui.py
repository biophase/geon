from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QToolButton, QMenu

from geon.ui.layers.pointcloud_ui import _ribbon
from geon.ui.scene_manager import SceneManager
from geon.rendering.scene import Scene
from test_pointcloud_blend import make_layer


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_ribbon_background_controls_and_render_coalescing(app):
    layer, _ = make_layer()
    viewer = SimpleNamespace(rerender=Mock())
    controller = SimpleNamespace(ctx=SimpleNamespace(viewer=viewer))
    parent = QWidget()
    ordinary = _ribbon(layer, parent, controller)
    assert ordinary.findChild(QWidget, "pointcloudBackground") is None
    layer.set_background_field_name("blue")
    ribbon = _ribbon(layer, parent, controller)
    column = ribbon.findChild(QWidget, "pointcloudBackground")
    slider = column.findChild(QSlider)
    assert slider.value() == 50
    for value in [20, 30, 40]:
        slider.setValue(value)
    assert viewer.rerender.call_count == 0
    app.processEvents()
    assert viewer.rerender.call_count == 1
    assert layer.foreground_mix == .4
    menu = column.findChild(QToolButton).menu()
    menu.aboutToShow.emit()
    assert [a.text() for a in menu.actions()] == ["None", "red", "blue"]
    menu.actions()[1].trigger()
    assert layer.background_field_name == "red"
    assert slider.value() == 40
    menu.actions()[0].trigger()
    assert column.isHidden()
    assert layer.background_field_name is None
    layer.set_background_field_name("blue")
    rebuilt = _ribbon(layer, parent, controller)
    assert rebuilt.findChild(QSlider).value() == 40
    rebuilt.findChild(QSlider).setValue(60)
    viewer.rerender.reset_mock()
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    viewer.rerender.assert_not_called()


def test_field_context_menu_targets_clicked_row(app, monkeypatch):
    layer, renderer = make_layer()
    other, _ = make_layer()
    scene = Scene(renderer)
    layer.detach()
    other.detach()
    layer = scene.add_data(layer.data)
    other = scene.add_data(other.data)
    layer.update()
    other.update()
    scene.active_layer_id = other.id
    viewer = SimpleNamespace(rerender=Mock())
    controller = SimpleNamespace(ctx=SimpleNamespace(viewer=viewer))
    manager = SceneManager(viewer, controller)
    manager._scene = scene
    manager.populate_tree()
    manager.show()
    app.processEvents()
    root = manager.tree.topLevelItem(0)
    root.child(0).setSelected(True)
    clicked = root.child(1)
    seen = []
    manager.broadcastLayerDisplayChanged.connect(seen.append)

    def execute(menu, *_):
        action = next(a for a in menu.actions() if a.text() == "Set as background field")
        action.trigger()

    monkeypatch.setattr(QMenu, "exec", execute)
    manager._on_tree_context_menu(manager.tree.visualItemRect(clicked).center())
    assert layer.background_field_name == "blue"
    assert layer.active_field_name == "red"
    assert scene.active_layer is other
    assert seen == [layer]
    viewer.rerender.assert_called_once()
    manager.close()
