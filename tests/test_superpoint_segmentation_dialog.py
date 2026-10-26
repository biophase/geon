import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication
from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.superpoint_segmentation_dialog import SuperpointSegmentationDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_selection_settings_and_layer_switch(qt_app):
    selected = PointCloudLayer(PointCloudData(np.zeros((4, 3), dtype=np.float32)))
    selected.active_selection = np.array([3, 1], dtype=np.int32)
    other = PointCloudLayer(PointCloudData(np.zeros((4, 3), dtype=np.float32)))
    scene = SimpleNamespace(layers={selected.id: selected, other.id: other})
    dialog = SuperpointSegmentationDialog(scene, selected, {"on_selection_only": True})
    assert dialog.on_selection_only()
    assert dialog.settings()["on_selection_only"] is True
    assert "on_selection_only" not in dialog.params()
    dialog.layer_combo.setCurrentIndex(1)
    assert not dialog.selection_only_box.isEnabled()
    assert not dialog.on_selection_only()
    dialog.layer_combo.setCurrentIndex(0)
    assert dialog.selection_only_box.isEnabled()
    assert not dialog.on_selection_only()
    dialog.close()


def test_saved_selection_mode_without_selection(qt_app):
    layer = PointCloudLayer(PointCloudData(np.zeros((4, 3), dtype=np.float32)))
    dialog = SuperpointSegmentationDialog(
        SimpleNamespace(layers={layer.id: layer}), layer, {"on_selection_only": True})
    assert not dialog.on_selection_only()
    assert not dialog.selection_only_box.isEnabled()
    dialog.close()


def test_selected_superpoint_features_stay_aligned(monkeypatch):
    from geon.algorithms import superpoints
    data = PointCloudData(np.arange(12, dtype=np.float32).reshape(4, 3))
    data.add_field("feature", np.array([[10], [20], [30], [40]], dtype=np.float32), FieldType.SCALAR)
    selection = np.array([3, 1], dtype=np.int32)
    subset, mapping = data.extract_subcloud(selection, field_names=["feature"])

    def segment(coords, features, **kwargs):
        np.testing.assert_array_equal(coords, data.points[selection])
        np.testing.assert_array_equal(features[:, 0], [40, 20])
        return np.array([0, 1], dtype=np.int32), {}

    monkeypatch.setattr(superpoints._native, "segment_superpoints", segment)
    labels, _ = superpoints.segment_superpoints(subset, feature_field_names=["feature"])
    output = np.full(len(data.points), -1, dtype=np.int32)
    output[mapping] = labels
    np.testing.assert_array_equal(output, [-1, 1, -1, 0])
