from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from geon.data.boundingbox import BoundingBoxData
from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.fit_obb_dialog import FitObbDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_fit_obb_dialog_lists_only_bounding_box_layers(qt_app):
    point_layer = PointCloudLayer(PointCloudData(np.zeros((3, 3), dtype=np.float32)))
    box_layer = BoundingBoxLayer(BoundingBoxData([]), browser_name="Targets")
    scene = SimpleNamespace(layers={point_layer.id: point_layer, box_layer.id: box_layer})

    dialog = FitObbDialog(scene, point_layer, 3)

    assert dialog.target_layer_combo.count() == 1
    assert dialog.target_layer() is box_layer
    assert dialog.method() == "pca"
    assert dialog.trim() == pytest.approx(0.01)


def test_fit_obb_dialog_disables_ok_without_target_layer(qt_app):
    point_layer = PointCloudLayer(PointCloudData(np.zeros((3, 3), dtype=np.float32)))
    scene = SimpleNamespace(layers={point_layer.id: point_layer})
    dialog = FitObbDialog(scene, point_layer, 3)
    buttons = dialog.findChild(QDialogButtonBox)

    assert buttons is not None
    assert not buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_fit_obb_dialog_offers_only_instance_and_semantic_fields(qt_app):
    data = PointCloudData(np.zeros((3, 3), dtype=np.float32))
    data.add_field("instances", np.asarray([0, 1, 1]), FieldType.INSTANCE)
    data.add_field("classes", np.asarray([2, 2, 3]), FieldType.SEMANTIC)
    data.add_field("values", np.zeros((3, 1), dtype=np.float32), FieldType.SCALAR)
    point_layer = PointCloudLayer(data)
    box_layer = BoundingBoxLayer(BoundingBoxData([]))
    scene = SimpleNamespace(layers={point_layer.id: point_layer, box_layer.id: box_layer})

    dialog = FitObbDialog(scene, point_layer, 3)

    assert [
        dialog.separate_field_combo.itemText(index)
        for index in range(dialog.separate_field_combo.count())
    ] == ["instances", "classes"]
    assert not dialog.separate_with_field()
    assert not dialog.separate_field_combo.isEnabled()
    dialog.separate_box.setChecked(True)
    assert dialog.separate_with_field()
    assert dialog.separation_field_name() == "instances"
