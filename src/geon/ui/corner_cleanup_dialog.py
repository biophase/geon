from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


CORNER_CLEANUP_DEFAULTS: dict[str, object] = {
    "source_field_name": None,
    "on_selection_only": False,
    "output_mode": "create_new",
    "output_field_base": "corner_cleaned_regions",
    "epsilon": 0.03,
    "neighbor_radius_factor": 3.0,
    "slenderness_threshold": 0.02,
    "absolute_width_factor": 3.0,
    "relative_width_factor": 0.5,
    "dual_support_threshold": 0.5,
    "min_corner_size": 10,
    "min_planar_size": 20,
}


class CornerCleanupDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        settings: Optional[dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clean up corner regions")
        self._layers: list[PointCloudLayer] = []
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.source_field_combo = QComboBox(self)
        form.addRow("Source instance field", self.source_field_combo)

        self.selection_only_box = QCheckBox("On selection only", self)
        form.addRow(self.selection_only_box)

        self.create_new_radio = QRadioButton("Create new field", self)
        self.update_source_radio = QRadioButton("Update source field", self)
        self.output_group = QButtonGroup(self)
        self.output_group.addButton(self.create_new_radio)
        self.output_group.addButton(self.update_source_radio)
        form.addRow(self.create_new_radio)

        self.output_field_edit = QLineEdit(self)
        form.addRow("New field base", self.output_field_edit)
        form.addRow(self.update_source_radio)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(1e-6, 1e6)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setSingleStep(0.01)
        form.addRow("epsilon", self.epsilon_spin)

        self.neighbor_factor_spin = QDoubleSpinBox(self)
        self.neighbor_factor_spin.setRange(0.1, 100.0)
        self.neighbor_factor_spin.setDecimals(3)
        self.neighbor_factor_spin.setSingleStep(0.5)
        form.addRow("Neighbor radius factor", self.neighbor_factor_spin)

        self.slenderness_spin = QDoubleSpinBox(self)
        self.slenderness_spin.setRange(0.0, 1.0)
        self.slenderness_spin.setDecimals(4)
        self.slenderness_spin.setSingleStep(0.005)
        self.slenderness_spin.setToolTip("RMS width divided by RMS length; lower is more slender.")
        form.addRow("Corner slenderness max", self.slenderness_spin)

        self.dual_support_spin = QDoubleSpinBox(self)
        self.dual_support_spin.setRange(0.0, 1.0)
        self.dual_support_spin.setDecimals(3)
        self.dual_support_spin.setSingleStep(0.05)
        form.addRow("Dual-neighbor support min", self.dual_support_spin)

        self.min_corner_size_spin = QSpinBox(self)
        self.min_corner_size_spin.setRange(3, 10_000_000)
        form.addRow("Minimum corner points", self.min_corner_size_spin)

        self.min_planar_size_spin = QSpinBox(self)
        self.min_planar_size_spin.setRange(3, 10_000_000)
        form.addRow("Minimum destination points", self.min_planar_size_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        self._populate_layers(scene, active_layer)
        self._refresh_layer_state()
        self._apply_settings(settings or {})
        self._update_output_state()
        self._validate()

        self.layer_combo.currentIndexChanged.connect(self._refresh_layer_state)
        self.source_field_combo.currentIndexChanged.connect(self._validate)
        self.selection_only_box.toggled.connect(self._validate)
        self.create_new_radio.toggled.connect(self._update_output_state)
        self.update_source_radio.toggled.connect(self._update_output_state)
        self.output_field_edit.textChanged.connect(self._validate)

    def _populate_layers(self, scene: Scene, active_layer: Optional[PointCloudLayer]) -> None:
        self._layers = [
            layer for layer in scene.layers.values()
            if isinstance(layer, PointCloudLayer)
        ]
        if not self._layers:
            self.layer_combo.addItem("<no point clouds>")
            self.layer_combo.setEnabled(False)
            return
        for layer in self._layers:
            self.layer_combo.addItem(layer.browser_name, layer)
        if active_layer is not None and active_layer in self._layers:
            self.layer_combo.setCurrentIndex(self._layers.index(active_layer))

    def _refresh_layer_state(self) -> None:
        self.source_field_combo.clear()
        layer = self.selected_layer()
        if layer is None:
            self.selection_only_box.setChecked(False)
            self.selection_only_box.setEnabled(False)
            self._validate()
            return
        for field in layer.data.get_fields(field_type=FieldType.INSTANCE):
            self.source_field_combo.addItem(field.name)
        if layer.active_field is not None and layer.active_field.field_type == FieldType.INSTANCE:
            index = self.source_field_combo.findText(layer.active_field.name)
            if index >= 0:
                self.source_field_combo.setCurrentIndex(index)
        selection = layer.active_selection
        has_selection = selection is not None and selection.size > 0
        self.selection_only_box.setEnabled(has_selection)
        if not has_selection:
            self.selection_only_box.setChecked(False)
        self._validate()

    def _apply_settings(self, settings: dict[str, object]) -> None:
        values = dict(CORNER_CLEANUP_DEFAULTS)
        values.update(settings)
        source_name = values.get("source_field_name")
        if isinstance(source_name, str):
            index = self.source_field_combo.findText(source_name)
            if index >= 0:
                self.source_field_combo.setCurrentIndex(index)
        self.selection_only_box.setChecked(
            bool(values["on_selection_only"]) and self.selection_only_box.isEnabled()
        )
        if values.get("output_mode") == "update_source":
            self.update_source_radio.setChecked(True)
        else:
            self.create_new_radio.setChecked(True)
        self.output_field_edit.setText(str(values["output_field_base"]))
        self.epsilon_spin.setValue(float(values["epsilon"]))
        self.neighbor_factor_spin.setValue(float(values["neighbor_radius_factor"]))
        self.slenderness_spin.setValue(float(values["slenderness_threshold"]))
        self.dual_support_spin.setValue(float(values["dual_support_threshold"]))
        self.min_corner_size_spin.setValue(int(values["min_corner_size"]))
        self.min_planar_size_spin.setValue(int(values["min_planar_size"]))

    def _update_output_state(self) -> None:
        self.output_field_edit.setEnabled(self.output_mode() == "create_new")
        self._validate()

    def _validate(self) -> None:
        valid = self.selected_layer() is not None and self.source_field_combo.count() > 0
        if self.output_mode() == "create_new":
            valid = valid and bool(self.output_field_base())
        if self._ok_button is not None:
            self._ok_button.setEnabled(valid)

    def selected_layer(self) -> Optional[PointCloudLayer]:
        layer = self.layer_combo.currentData()
        return layer if isinstance(layer, PointCloudLayer) else None

    def source_field_name(self) -> Optional[str]:
        name = self.source_field_combo.currentText().strip()
        return name or None

    def on_selection_only(self) -> bool:
        return self.selection_only_box.isEnabled() and self.selection_only_box.isChecked()

    def output_mode(self) -> str:
        return "update_source" if self.update_source_radio.isChecked() else "create_new"

    def output_field_base(self) -> str:
        return self.output_field_edit.text().strip()

    def params(self) -> dict[str, object]:
        return {
            "epsilon": float(self.epsilon_spin.value()),
            "neighbor_radius_factor": float(self.neighbor_factor_spin.value()),
            "slenderness_threshold": float(self.slenderness_spin.value()),
            "absolute_width_factor": float(CORNER_CLEANUP_DEFAULTS["absolute_width_factor"]),
            "relative_width_factor": float(CORNER_CLEANUP_DEFAULTS["relative_width_factor"]),
            "dual_support_threshold": float(self.dual_support_spin.value()),
            "min_corner_size": int(self.min_corner_size_spin.value()),
            "min_planar_size": int(self.min_planar_size_spin.value()),
        }

    def settings(self) -> dict[str, object]:
        result = self.params()
        result.update({
            "source_field_name": self.source_field_name(),
            "on_selection_only": self.on_selection_only(),
            "output_mode": self.output_mode(),
            "output_field_base": self.output_field_base(),
        })
        return result
