from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


SUPERPOINT_DEFAULTS: dict[str, object] = {
    "k_neighbors": 10,
    "regularization": 0.05,
    "spatial_weight": 1.0,
    "cutoff": 10,
    "iterations": 10,
    "parallel": True,
    "output_field_base": "superpoints",
}


def _field_supported(field_type: FieldType) -> bool:
    return field_type in {
        FieldType.SCALAR,
        FieldType.VECTOR,
        FieldType.NORMAL,
        FieldType.INTENSITY,
    }


class SuperpointSegmentationDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        settings: Optional[dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Superpoint segmentation")

        self._layers: list[PointCloudLayer] = []
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        feature_group = QGroupBox("Optional feature fields", self)
        feature_layout = QVBoxLayout(feature_group)
        self.feature_list = QListWidget(feature_group)
        self.feature_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.feature_list.setMinimumHeight(140)
        feature_layout.addWidget(self.feature_list)
        helper = QLabel(
            "Scalar, vector, normal, and intensity fields are supported. "
            "Coordinates are always included automatically.",
            feature_group,
        )
        helper.setWordWrap(True)
        feature_layout.addWidget(helper)
        layout.addWidget(feature_group)

        self.k_neighbors_spin = QSpinBox(self)
        self.k_neighbors_spin.setRange(1, 256)
        form.addRow("k neighbors", self.k_neighbors_spin)

        self.regularization_spin = QDoubleSpinBox(self)
        self.regularization_spin.setRange(1e-6, 1e6)
        self.regularization_spin.setDecimals(6)
        self.regularization_spin.setSingleStep(0.01)
        form.addRow("regularization", self.regularization_spin)

        self.spatial_weight_spin = QDoubleSpinBox(self)
        self.spatial_weight_spin.setRange(1e-6, 1e6)
        self.spatial_weight_spin.setDecimals(6)
        self.spatial_weight_spin.setSingleStep(0.1)
        form.addRow("spatial weight", self.spatial_weight_spin)

        self.cutoff_spin = QSpinBox(self)
        self.cutoff_spin.setRange(1, 10_000_000)
        form.addRow("cutoff", self.cutoff_spin)

        self.iterations_spin = QSpinBox(self)
        self.iterations_spin.setRange(1, 1000)
        form.addRow("iterations", self.iterations_spin)

        flags_row = QWidget(self)
        flags_layout = QHBoxLayout(flags_row)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        self.parallel_box = QCheckBox("Parallel", flags_row)
        flags_layout.addWidget(self.parallel_box)
        flags_layout.addStretch(1)
        form.addRow("Execution", flags_row)

        self.output_field_edit = QLineEdit(self)
        form.addRow("Output field base", self.output_field_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        self._populate_layers(scene, active_layer)
        self._refresh_feature_list()
        self._apply_settings(settings or {})
        self._validate()

        self.layer_combo.currentIndexChanged.connect(self._refresh_feature_list)
        self.output_field_edit.textChanged.connect(self._validate)

    def _apply_settings(self, settings: dict[str, object]) -> None:
        merged = dict(SUPERPOINT_DEFAULTS)
        merged.update(settings)
        self.k_neighbors_spin.setValue(int(merged["k_neighbors"]))
        self.regularization_spin.setValue(float(merged["regularization"]))
        self.spatial_weight_spin.setValue(float(merged["spatial_weight"]))
        self.cutoff_spin.setValue(int(merged["cutoff"]))
        self.iterations_spin.setValue(int(merged["iterations"]))
        self.parallel_box.setChecked(bool(merged["parallel"]))
        self.output_field_edit.setText(str(merged["output_field_base"]))
        selected_fields = merged.get("feature_field_names")
        if isinstance(selected_fields, list):
            selected_names = {str(name) for name in selected_fields}
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                item.setSelected(isinstance(item.data(0x0100), str) and item.data(0x0100) in selected_names)

    def _populate_layers(self, scene: Scene, active_layer: Optional[PointCloudLayer]) -> None:
        self._layers = [
            layer for layer in scene.layers.values()
            if isinstance(layer, PointCloudLayer)
        ]
        self.layer_combo.clear()
        if not self._layers:
            self.layer_combo.addItem("<no point clouds>")
            self.layer_combo.setEnabled(False)
            return
        for layer in self._layers:
            self.layer_combo.addItem(layer.browser_name, layer)
        if active_layer is not None and active_layer in self._layers:
            self.layer_combo.setCurrentIndex(self._layers.index(active_layer))

    def _refresh_feature_list(self) -> None:
        self.feature_list.clear()
        layer = self.selected_layer()
        if layer is None:
            self._validate()
            return
        for field in layer.data.get_fields():
            if not _field_supported(field.field_type):
                continue
            text = f"{field.name} ({field.field_type.human_name})"
            item = QListWidgetItem(text, self.feature_list)
            item.setData(0x0100, field.name)
        self._validate()

    def _validate(self) -> None:
        ok = self.selected_layer() is not None and bool(self.output_field_base())
        if self._ok_button is not None:
            self._ok_button.setEnabled(ok)

    def selected_layer(self) -> Optional[PointCloudLayer]:
        layer = self.layer_combo.currentData()
        return layer if isinstance(layer, PointCloudLayer) else None

    def selected_feature_field_names(self) -> list[str]:
        names: list[str] = []
        for item in self.feature_list.selectedItems():
            value = item.data(0x0100)
            if isinstance(value, str):
                names.append(value)
        return names

    def output_field_base(self) -> str:
        return self.output_field_edit.text().strip()

    def params(self) -> dict[str, object]:
        return {
            "k_neighbors": int(self.k_neighbors_spin.value()),
            "regularization": float(self.regularization_spin.value()),
            "spatial_weight": float(self.spatial_weight_spin.value()),
            "cutoff": int(self.cutoff_spin.value()),
            "iterations": int(self.iterations_spin.value()),
            "parallel": bool(self.parallel_box.isChecked()),
        }

    def settings(self) -> dict[str, object]:
        out = self.params()
        out["feature_field_names"] = self.selected_feature_field_names()
        out["output_field_base"] = self.output_field_base()
        return out
