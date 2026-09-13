from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.connected_components_dialog import ConnectedComponentsDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _layer(*, selected: bool, with_instance: bool = True) -> PointCloudLayer:
    data = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    if with_instance:
        data.add_field(
            name="instances",
            data=np.asarray([0, 0, 2, -1], dtype=np.int32),
            field_type=FieldType.INSTANCE,
        )
    layer = PointCloudLayer(data)
    if selected:
        layer.active_selection = np.asarray([1, 2], dtype=np.int32)
    return layer


def test_dialog_restores_selection_and_existing_output(qt_app):
    layer = _layer(selected=True)
    dialog = ConnectedComponentsDialog(
        SimpleNamespace(layers={layer.id: layer}),
        layer,
        settings={
            "epsilon": 0.08,
            "on_selection_only": True,
            "output_mode": "write_existing",
            "output_existing_field_name": "instances",
        },
    )

    assert dialog.epsilon() == pytest.approx(0.08)
    assert dialog.on_selection_only()
    assert dialog.output_mode() == "write_existing"
    assert dialog.existing_field_name() == "instances"
    assert not dialog.output_field_edit.isEnabled()


def test_dialog_disables_unavailable_modes(qt_app):
    layer = _layer(selected=False, with_instance=False)
    dialog = ConnectedComponentsDialog(SimpleNamespace(layers={layer.id: layer}), layer)

    assert not dialog.selection_only_box.isEnabled()
    assert not dialog.write_existing_radio.isEnabled()
    assert dialog.output_mode() == "create_new"


def test_dialog_filters_existing_output_to_instance_fields(qt_app):
    layer = _layer(selected=False)
    layer.data.add_field(
        name="scalar",
        data=np.zeros((4, 1), dtype=np.float32),
        field_type=FieldType.SCALAR,
    )
    dialog = ConnectedComponentsDialog(SimpleNamespace(layers={layer.id: layer}), layer)

    assert dialog.existing_field_combo.count() == 1
    assert dialog.existing_field_combo.itemText(0) == "instances"
