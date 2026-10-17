from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QToolButton, QMenu, QLabel, QFileDialog, QComboBox
from PyQt6.QtCore import Qt
import numpy as np
import json

from geon.data.pointcloud import FieldType
from geon.config.theme import UIStyle

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
    label = column.findChild(QLabel)
    assert label.text() == "Background: "
    assert label.styleSheet() == UIStyle.TYPE_LABEL.value
    combo = column.findChild(QComboBox)
    assert combo.currentText() == "blue"
    assert combo.sizeHint().height() == QComboBox().sizeHint().height()
    slider = column.findChild(QSlider)
    assert slider.value() == 50
    for value in [20, 30, 40]:
        slider.setValue(value)
    assert viewer.rerender.call_count == 0
    app.processEvents()
    assert viewer.rerender.call_count == 1
    assert layer.foreground_mix == .4
    assert [combo.itemText(i) for i in range(combo.count())] == ["None", "red", "blue"]
    combo.setCurrentIndex(1)
    assert layer.background_field_name == "red"
    assert combo.currentText() == "red"
    assert slider.value() == 40
    combo.setCurrentIndex(0)
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


def test_semantic_export_from_ribbon_and_clicked_field(app, monkeypatch, tmp_path):
    layer, renderer = make_layer()
    layer.data.add_field('classes', np.zeros(201, dtype=np.int32), FieldType.SEMANTIC)
    field = layer.data.get_fields('classes')[0]
    layer.active_field_name = 'classes'
    layer.set_background_field_name('blue')
    controller = SimpleNamespace(ctx=SimpleNamespace(viewer=SimpleNamespace(rerender=Mock())))
    parent = QWidget()
    ribbon = _ribbon(layer, parent, controller)
    ribbon.show()
    app.processEvents()
    more = ribbon.findChild(QToolButton, 'pointcloudMore')
    background = ribbon.findChild(QWidget, 'pointcloudBackground')
    assert ribbon.layout().indexOf(more) > ribbon.layout().indexOf(background)
    assert ribbon.layout().itemAt(ribbon.layout().indexOf(more)).alignment() & Qt.AlignmentFlag.AlignBottom
    path = tmp_path / 'schema.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    more.menu().actions()[0].trigger()
    assert json.loads(path.read_text()) == json.loads(json.dumps(field.schema.to_dict()))
    path.unlink()

    layer.detach()
    scene = Scene(renderer)
    layer = scene.add_data(layer.data)
    layer.active_field_name = 'red'  # Export must use the clicked field, not the active one.
    manager = SceneManager(controller.ctx.viewer, controller)
    manager._scene = scene
    manager.populate_tree()
    manager.show()
    app.processEvents()
    root = manager.tree.topLevelItem(0)
    clicked = next(root.child(i) for i in range(root.childCount())
                   if root.child(i).data(0, Qt.ItemDataRole.UserRole) is field)

    def execute(menu, *_):
        next(a for a in menu.actions() if a.text() == 'Export to JSON').trigger()

    monkeypatch.setattr(QMenu, 'exec', execute)
    manager._on_tree_context_menu(manager.tree.visualItemRect(clicked).center())
    assert json.loads(path.read_text()) == json.loads(json.dumps(field.schema.to_dict()))
    manager.close()
    ribbon.close()
    parent.close()
