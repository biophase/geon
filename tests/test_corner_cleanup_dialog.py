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
from geon.ui.corner_cleanup_dialog import CornerCleanupDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _layer(*, selected: bool) -> PointCloudLayer:
    data = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    data.add_field(
        name="regions",
        data=np.asarray([0, 0, 1, 2], dtype=np.int32),
        field_type=FieldType.INSTANCE,
    )
    layer = PointCloudLayer(data)
    if selected:
        layer.active_selection = np.asarray([1, 2, 3], dtype=np.int32)
    return layer


def test_corner_cleanup_dialog_restores_selection_and_output_mode(qt_app):
    layer = _layer(selected=True)
    dialog = CornerCleanupDialog(
        SimpleNamespace(layers={layer.id: layer}),
        layer,
        settings={
            "source_field_name": "regions",
            "on_selection_only": True,
            "output_mode": "update_source",
        },
    )

    assert dialog.source_field_name() == "regions"
    assert dialog.on_selection_only()
    assert dialog.output_mode() == "update_source"
    assert not dialog.output_field_edit.isEnabled()


def test_corner_cleanup_selection_is_unavailable_without_selection(qt_app):
    layer = _layer(selected=False)
    dialog = CornerCleanupDialog(SimpleNamespace(layers={layer.id: layer}), layer)

    assert not dialog.selection_only_box.isEnabled()
    assert not dialog.on_selection_only()


def test_corner_cleanup_uses_new_geometry_defaults_and_ignores_legacy_values(qt_app):
    layer = _layer(selected=False)
    dialog = CornerCleanupDialog(
        SimpleNamespace(layers={layer.id: layer}),
        layer,
        settings={"linearity_threshold": 0.4, "planarity_threshold": 0.4},
    )

    assert dialog.params()["slenderness_threshold"] == pytest.approx(0.02)
    assert dialog.params()["absolute_width_factor"] == pytest.approx(3.0)
    assert dialog.params()["relative_width_factor"] == pytest.approx(0.5)
    assert "linearity_threshold" not in dialog.params()
    assert "planarity_threshold" not in dialog.params()
