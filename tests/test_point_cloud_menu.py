from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

from geon.data.boundingbox import BoundingBoxData
from geon.data.cellcomplex import CellComplexData
from geon.data.pointcloud import PointCloudData
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.layer_checks import require_active_layer
from geon.ui.main_window import MainWindow
from geon.ui.menu_bar import MenuBar


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qt_app):
    # Test real menu wiring without initializing a graphics context.
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window.scene_manager = SimpleNamespace(_scene=None)
    window.menu_bar = MenuBar(window)
    window._populate_point_cloud_menu()
    yield window
    window.scene_manager._scene = None
    window.close()


def test_document_menu_groups_point_cloud_tools(window):
    bar = window.menu_bar
    document = {a.text(): a for a in bar.doc_menu.actions()}
    assert document["Point cloud"].menu() is bar.point_cloud_menu
    assert document["Bounding box"].menu() is bar.bounding_box_menu
    assert document["Cell Complex"].menu() is bar.cell_complex_menu
    assert bar.bounding_box_menu.isEmpty()
    assert bar.cell_complex_menu.isEmpty()
    assert [a.text() for a in bar.point_cloud_menu.actions()] == [
        "Import field from ...", "Edit fields", "Compute geometric features", "Segmentation"
    ]
    for action in bar.point_cloud_menu.actions():
        assert action.text() not in document
    segmentation = bar.point_cloud_menu.actions()[-1].menu()
    assert len(segmentation.actions()) == 6


def point_cloud_actions(window):
    for action in window.menu_bar.point_cloud_menu.actions():
        if action.menu() is not None:
            yield from action.menu().actions()
        else:
            yield action


@pytest.mark.parametrize("scene_kind", ["no_scene", "no_active", "bounding_box", "cell_complex"])
def test_each_point_cloud_action_checks_active_layer(window, monkeypatch, scene_kind):
    layers = {
        "bounding_box": BoundingBoxLayer(BoundingBoxData()),
        "cell_complex": CellComplexLayer(CellComplexData()),
    }
    window.scene_manager._scene = (
        None if scene_kind == "no_scene" else SimpleNamespace(active_layer=layers.get(scene_kind))
    )
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: errors.append(text))
    actions = list(point_cloud_actions(window))
    assert len(actions) == 9
    for action in actions:
        action.trigger()
    assert errors == ["Activate a point cloud first."] * len(actions)
    window.scene_manager._scene = None


@pytest.mark.parametrize("layer_type, data_factory, name", [
    (PointCloudLayer, lambda: PointCloudData(np.zeros((1, 3), dtype=np.float32)), "point cloud"),
    (BoundingBoxLayer, BoundingBoxData, "bounding box"),
    (CellComplexLayer, CellComplexData, "cell complex"),
])
def test_reusable_layer_check(qt_app, monkeypatch, layer_type, data_factory, name):
    parent = QWidget()
    layer = layer_type(data_factory())
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: errors.append(text))
    assert require_active_layer(parent, SimpleNamespace(active_layer=layer), layer_type) is layer
    assert errors == []
    assert require_active_layer(parent, None, layer_type) is None
    assert errors == [f"Activate a {name} first."]
