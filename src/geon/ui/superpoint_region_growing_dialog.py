"""UI adapter for the removable Python superpoint growth experiment."""
from __future__ import annotations

from threading import Event
from time import perf_counter

import numpy as np
from PyQt6.QtCore import QEventLoop, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QProgressDialog,
    QSpinBox, QVBoxLayout,
)

from ..algorithms.superpoint_region_growing import (
    DEFAULT_PARAMS, SuperpointGrowthCancelled, segment_planar_superpoints,
)
from ..data.pointcloud import FieldType
from ..io.dataset import RefModState
from ..rendering.pointcloud import PointCloudLayer
from .layer_checks import require_active_layer


TITLE = "Planar region growing on superpoints (experimental)"
OUTPUT_BASE = "superpoint_planar_regions"


class SuperpointRegionGrowingDialog(QDialog):
    def __init__(self, scene, active_layer, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.layer_combo = QComboBox(self)
        self.instance_field_combo = QComboBox(self)
        self.selection_only_box = QCheckBox("On selection only", self)
        form.addRow("Point cloud layer", self.layer_combo)
        form.addRow("Source instance field", self.instance_field_combo)
        form.addRow(self.selection_only_box)
        hint = QLabel(
            "Each instance is one superpoint. Selecting any point includes its entire "
            "superpoint. Normals are fitted from member points; distance tests use "
            "superpoint centroids. Distances use the cloud's coordinate units.", self)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        advanced = QGroupBox("Advanced growth and seed settings", self)
        advanced_form = QFormLayout(advanced)
        self.parameter_widgets = {}
        specifications = [
            ("min_d", "Touching distance (min_d)", 0, 1e6, False),
            ("epsilon", "Plane scale (epsilon)", 1e-6, 1e6, False),
            ("epsilon_multiplier", "Plane tolerance multiplier", 1e-6, 1e6, False),
            ("alpha_deg", "Normal deviation (degrees)", 0.000001, 90, False),
            ("tau", "Minimum region size (points)", 1, 1000000000, False),
            ("first_refit", "First refit (superpoints)", 1, 1000000000, True),
            ("refit_multiplier", "Refit interval multiplier", 1.000001, 1e6, True),
            ("max_dist_from_cent", "Angle tightening distance", 1e-6, 1e6, True),
            ("seed_min_points", "Minimum seed size (points)", 1, 1000000000, True),
            ("seed_planarity_min", "Minimum seed planarity", 0, 1, True),
            ("seed_scattering_max", "Maximum seed scattering", 0, 1, True),
        ]
        for key, label, minimum, maximum, is_advanced in specifications:
            default = DEFAULT_PARAMS[key]
            widget = QSpinBox(self) if isinstance(default, int) else QDoubleSpinBox(self)
            if isinstance(widget, QDoubleSpinBox):
                widget.setDecimals(6)
                widget.setSingleStep(1.0 if key == "alpha_deg" else 0.01)
            widget.setRange(minimum, maximum)
            widget.setValue(settings.get(key, default))
            self.parameter_widgets[key] = widget
            (advanced_form if is_advanced else form).addRow(label, widget)
        self.seed_gating_box = QCheckBox("Enable seed gating", self)
        self.seed_gating_box.setChecked(bool(settings.get("enable_seed_gating", True)))
        advanced_form.insertRow(0, self.seed_gating_box)
        layout.addWidget(advanced)
        output_form = QFormLayout()
        self.output_field_edit = QLineEdit(str(settings.get("output_field_base", OUTPUT_BASE)), self)
        output_form.addRow("New output field base", self.output_field_edit)
        layout.addLayout(output_form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        for layer in scene.layers.values():
            if isinstance(layer, PointCloudLayer):
                self.layer_combo.addItem(layer.data.name, layer)
        for i in range(self.layer_combo.count()):
            if self.layer_combo.itemData(i) is active_layer:
                self.layer_combo.setCurrentIndex(i)
                break
        self.layer_combo.currentIndexChanged.connect(self._refresh_fields)
        self.instance_field_combo.currentIndexChanged.connect(self._validate)
        self.output_field_edit.textChanged.connect(self._validate)
        self._refresh_fields()
        self.selection_only_box.setChecked(
            self.selection_only_box.isEnabled() and bool(settings.get("on_selection_only", False)))

    def selected_layer(self):
        return self.layer_combo.currentData()

    def source_field_name(self):
        return self.instance_field_combo.currentText() or None

    def output_field_base(self):
        return self.output_field_edit.text().strip()

    def on_selection_only(self):
        return self.selection_only_box.isEnabled() and self.selection_only_box.isChecked()

    def _refresh_fields(self):
        self.instance_field_combo.clear()
        layer = self.selected_layer()
        if layer is not None:
            for field in layer.data.get_fields(field_type=FieldType.INSTANCE):
                self.instance_field_combo.addItem(field.name)
            if layer.active_field is not None:
                index = self.instance_field_combo.findText(layer.active_field.name)
                if index >= 0:
                    self.instance_field_combo.setCurrentIndex(index)
        selection = layer.active_selection if layer is not None else None
        enabled = selection is not None and selection.size > 0
        self.selection_only_box.setEnabled(enabled)
        if not enabled:
            self.selection_only_box.setChecked(False)
        self._validate()

    def _validate(self):
        self._ok_button.setEnabled(
            self.selected_layer() is not None and bool(self.source_field_name())
            and bool(self.output_field_base()))

    def params(self):
        params = {key: widget.value() for key, widget in self.parameter_widgets.items()}
        params["enable_seed_gating"] = self.seed_gating_box.isChecked()
        return params

    def settings(self):
        return dict(self.params(), output_field_base=self.output_field_base(),
                    on_selection_only=self.on_selection_only())


class _SuperpointGrowthWorker(QThread):
    progressed = pyqtSignal(str, int, int)

    def __init__(self, points, labels, selection, params, parent=None):
        super().__init__(parent)
        self.points, self.labels = points, labels
        self.selection, self.params = selection, params
        self.cancel_event = Event()
        self.result = None
        self.error = None

    def run(self):
        last_time = 0.0
        last_stage = None

        def progress(stage, done, total):
            nonlocal last_time, last_stage
            now = perf_counter()
            if stage != last_stage or now - last_time >= 0.1 or done == total:
                self.progressed.emit(stage, done, total)
                last_time, last_stage = now, stage

        try:
            self.result = segment_planar_superpoints(
                self.points, instance_labels=self.labels, selection_indices=self.selection,
                params=self.params, progress_callback=progress,
                cancel_requested=self.cancel_event.is_set)
        except SuperpointGrowthCancelled:
            self.cancel_event.set()
        except Exception as exc:
            self.error = str(exc)


def _apply_result(window, scene, doc, layer, data, source_field, worker, output_base):
    """Commit a completed snapshot on the GUI thread only if its target is current."""
    if worker.cancel_event.is_set() or worker.error is not None or worker.result is None:
        return False
    if (window.scene_manager._scene is not scene or scene.doc is not doc
            or scene.layers.get(layer.id) is not layer or layer.data is not data):
        return False
    fields = data.get_fields(names=source_field.name)
    if (not fields or fields[0] is not source_field
            or not np.array_equal(data.points, worker.points)
            or not np.array_equal(source_field.data, worker.labels)):
        return False
    labels, stats = worker.result
    if labels.shape != (len(data.points),):
        raise ValueError("Output label length does not match point count")
    name = window._unique_field_name(data.field_names, output_base)
    data.add_field(name=name, data=labels[:, None], field_type=FieldType.INSTANCE)
    scene.active_layer_id = layer.id
    layer.set_active_field_name(name)
    layer.update()
    window.scene_manager.broadcastActivatedLayer.emit(layer)
    window.scene_manager.broadcastActivatedPcdField.emit(layer)
    dataset = window.dataset_manager._dataset
    if dataset is not None:
        for ref in dataset.doc_refs:
            if ref.name == scene.doc.name:
                ref.modState = RefModState.MODIFIED
                break
    window.scene_manager.populate_tree()
    window.viewer.rerender()
    print(f"[superpoint_region_growing] Added '{name}': {stats}")
    return True


def run_superpoint_region_growing(window):
    scene = window.scene_manager._scene
    active_layer = require_active_layer(window, scene, PointCloudLayer)
    if active_layer is None:
        return
    dialog = SuperpointRegionGrowingDialog(
        scene, active_layer, window.preferences.get_superpoint_region_growing_settings(), window)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    layer = dialog.selected_layer()
    if layer is None:
        return
    data, doc = layer.data, scene.doc
    fields = data.get_fields(names=dialog.source_field_name(), field_type=FieldType.INSTANCE)
    if not fields:
        QMessageBox.warning(window, TITLE, "The source instance field is no longer available.")
        return
    selection = None
    if dialog.on_selection_only():
        if layer.active_selection is None or not layer.active_selection.size:
            QMessageBox.warning(window, TITLE, "The point cloud no longer has a selection.")
            return
        selection = layer.active_selection.copy()
    window.preferences.set_superpoint_region_growing_settings(dialog.settings())
    window.preferences.save()
    source_field = fields[0]
    worker = _SuperpointGrowthWorker(
        data.points.copy(), source_field.data.copy(), selection, dialog.params(), window)
    progress_dialog = QProgressDialog("Preparing superpoints...", "Cancel", 0, 0, window)
    progress_dialog.setWindowTitle(TITLE)
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setAutoClose(False)
    progress_dialog.setAutoReset(False)

    def cancel():
        worker.cancel_event.set()
        progress_dialog.setLabelText("Cancelling...")

    def progress(stage, done, total):
        if worker.cancel_event.is_set():
            return
        progress_dialog.setLabelText(stage)
        progress_dialog.setMaximum(total if total else 0)
        progress_dialog.setValue(done)

    progress_dialog.canceled.connect(cancel)
    worker.progressed.connect(progress)
    loop = QEventLoop(window)
    worker.finished.connect(loop.quit)
    progress_dialog.show()
    worker.start()
    loop.exec()
    worker.wait()
    progress_dialog.canceled.disconnect(cancel)
    progress_dialog.close()
    try:
        if worker.error is not None and not worker.cancel_event.is_set():
            QMessageBox.critical(window, TITLE, worker.error)
        elif not worker.cancel_event.is_set():
            if not _apply_result(window, scene, doc, layer, data, source_field, worker, dialog.output_field_base()):
                QMessageBox.information(window, TITLE, "The target cloud changed. The result was discarded.")
    except Exception as exc:
        QMessageBox.critical(window, TITLE, f"Could not write segmentation output: {exc}")
    finally:
        worker.deleteLater()
        progress_dialog.deleteLater()
