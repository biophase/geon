import os
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.settings import Preferences
from geon.ui import superpoint_region_growing_dialog as ui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def layer_with_fields():
    a, b = np.meshgrid(np.arange(4) * .1, np.arange(4) * .1)
    data = PointCloudData(np.column_stack((a.ravel(), b.ravel(), np.zeros(16))))
    data.add_field("scalar", np.zeros((16, 1)), FieldType.SCALAR)
    data.add_field("first", np.zeros((16, 1), dtype=np.int32), FieldType.INSTANCE)
    data.add_field("active", np.ones((16, 1), dtype=np.int32), FieldType.INSTANCE)
    layer = PointCloudLayer(data)
    layer.set_active_field_name("active")
    return layer


def test_dialog_filters_fields_and_selection_and_roundtrips(app):
    layer = layer_with_fields()
    layer.active_selection = np.array([2], dtype=np.int32)
    other = PointCloudLayer(PointCloudData(np.zeros((1, 3))))
    scene = SimpleNamespace(layers={layer.id: layer, other.id: other})
    dialog = ui.SuperpointRegionGrowingDialog(scene, layer, {"on_selection_only": True, "tau": 3})
    assert dialog.instance_field_combo.count() == 2
    assert dialog.source_field_name() == "active"
    assert dialog.on_selection_only()
    assert dialog._ok_button.isEnabled()
    dialog.parameter_widgets["min_d"].setValue(.125)
    restored = ui.SuperpointRegionGrowingDialog(scene, layer, dialog.settings())
    assert restored.params() == dialog.params()
    assert restored.on_selection_only()
    dialog.layer_combo.setCurrentIndex(1)
    assert not dialog._ok_button.isEnabled()
    assert not dialog.selection_only_box.isEnabled()
    assert not dialog.on_selection_only()
    dialog.layer_combo.setCurrentIndex(0)
    assert dialog._ok_button.isEnabled()
    dialog.output_field_edit.setText("   ")
    assert not dialog._ok_button.isEnabled()
    restored.close()
    dialog.close()


def test_defaults_fall_back_to_first_instance_and_empty_scene(app):
    layer = layer_with_fields()
    layer.set_active_field_name("scalar")
    dialog = ui.SuperpointRegionGrowingDialog(SimpleNamespace(layers={layer.id: layer}), layer)
    assert dialog.source_field_name() == "first"
    assert not dialog.on_selection_only()
    assert dialog.params() == ui.DEFAULT_PARAMS
    dialog.close()
    dialog = ui.SuperpointRegionGrowingDialog(SimpleNamespace(layers={}), None)
    assert not dialog._ok_button.isEnabled()
    dialog.close()


def test_preferences_round_trip_and_legacy_file(tmp_path):
    path = tmp_path / "preferences.toml"
    path.write_text('user_name = "Example"\n', encoding="utf-8")
    prefs = Preferences.load(path)
    assert prefs.get_superpoint_region_growing_settings() == {}
    values = dict(ui.DEFAULT_PARAMS, output_field_base="experiment", on_selection_only=True)
    prefs.set_superpoint_region_growing_settings(values)
    prefs.save()
    loaded = Preferences.load(path)
    assert loaded.get_superpoint_region_growing_settings() == values
    assert "superpoint_region_growing__min_d" in path.read_text()
    assert loaded.get_region_growing_settings() == {}


def test_worker_success_cancel_and_failure(app):
    layer = layer_with_fields()
    field = layer.data.get_fields(names="active")[0]
    worker = ui._SuperpointGrowthWorker(layer.data.points.copy(), field.data.copy(), None, {"tau": 3})
    worker.start()
    assert worker.wait(10000)
    assert worker.error is None
    assert worker.result is not None
    assert np.all(worker.result[0] == 0)
    cancelled = ui._SuperpointGrowthWorker(layer.data.points, field.data, None, {})
    cancelled.cancel_event.set()
    cancelled.run()
    assert cancelled.result is None and cancelled.error is None
    failed = ui._SuperpointGrowthWorker(layer.data.points, np.zeros(2), None, {})
    failed.run()
    assert failed.result is None and failed.error is not None


def commit_fixture():
    layer = layer_with_fields()
    field = layer.data.get_fields(names="active")[0]
    data = layer.data
    scene = SimpleNamespace(doc=SimpleNamespace(name="doc"), layers={layer.id: layer})
    calls = []
    layer.update = lambda: calls.append("update")
    signal = SimpleNamespace(emit=lambda value: calls.append("emit"))
    reference = SimpleNamespace(name="doc", modState=None)
    window = SimpleNamespace(
        scene_manager=SimpleNamespace(_scene=scene, broadcastActivatedLayer=signal,
            broadcastActivatedPcdField=signal, populate_tree=lambda: calls.append("tree")),
        dataset_manager=SimpleNamespace(_dataset=SimpleNamespace(doc_refs=[reference])),
        viewer=SimpleNamespace(rerender=lambda: calls.append("render")),
        _unique_field_name=lambda names, base: base if base not in names else base + "_001")
    worker = SimpleNamespace(cancel_event=Event(), error=None,
        result=(np.zeros(len(data.points), dtype=np.int32), {}),
        points=data.points.copy(), labels=field.data.copy())
    return window, scene, layer, data, field, worker, calls, reference


def test_commit_creates_new_field_preserves_source_and_marks_modified(app):
    window, scene, layer, data, field, worker, calls, reference = commit_fixture()
    assert ui._apply_result(window, scene, scene.doc, layer, data, field, worker, "active")
    assert layer.active_field.name == "active_001"
    assert data.get_fields(names="active_001")[0].field_type == FieldType.INSTANCE
    np.testing.assert_array_equal(field.data, worker.labels)
    assert "render" in calls and "tree" in calls
    assert reference.modState == ui.RefModState.MODIFIED


@pytest.mark.parametrize("change", ["cancel", "failure", "scene", "doc", "layer", "data", "points", "labels", "field"])
def test_commit_discards_cancelled_failed_or_stale_results(app, change):
    window, scene, layer, data, field, worker, calls, reference = commit_fixture()
    doc = scene.doc
    if change == "cancel":
        worker.cancel_event.set()
    elif change == "failure":
        worker.error = "Failed"
    elif change == "scene":
        window.scene_manager._scene = None
    elif change == "doc":
        scene.doc = SimpleNamespace(name="replacement")
    elif change == "layer":
        scene.layers.clear()
    elif change == "data":
        layer.data = PointCloudData(np.zeros((16, 3)))
    elif change == "points":
        data.points[0, 0] += 1
    elif change == "labels":
        field.data[0, 0] = 99
    elif change == "field":
        field.name = "removed"
        data._fields.remove(field)
    assert not ui._apply_result(window, scene, doc, layer, data, field, worker, "result")
    assert "result" not in data.field_names
    assert not calls and reference.modState is None


@pytest.mark.parametrize("cancelled", [False, True])
def test_action_runs_worker_and_commits_only_on_success(app, tmp_path, monkeypatch, cancelled):
    setup, scene, layer, data, field, _, calls, reference = commit_fixture()
    window = QWidget()
    window.scene_manager = setup.scene_manager
    window.dataset_manager = setup.dataset_manager
    window.viewer = setup.viewer
    window._unique_field_name = setup._unique_field_name
    window.preferences = Preferences(path=tmp_path / "prefs.toml")
    window.preferences.set_superpoint_region_growing_settings({"tau": 3})
    scene.active_layer = layer
    monkeypatch.setattr(ui.SuperpointRegionGrowingDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    original_start = ui._SuperpointGrowthWorker.start
    if cancelled:
        def start(self):
            self.cancel_event.set()
            original_start(self)
        monkeypatch.setattr(ui._SuperpointGrowthWorker, "start", start)
    ui.run_superpoint_region_growing(window)
    assert (ui.OUTPUT_BASE in data.field_names) is not cancelled
    assert bool(calls) is not cancelled
    assert window.preferences.get_superpoint_region_growing_settings()["tau"] == 3
    window.close()
