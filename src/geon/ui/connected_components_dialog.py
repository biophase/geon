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
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene
from .segmentation_confirmation import confirm_global_field_overwrite


CONNECTED_COMPONENTS_DEFAULTS: dict[str, object] = {
    "epsilon": 0.03,
    "on_selection_only": False,
    "output_mode": "create_new",
    "output_field_base": "connected_components",
    "output_existing_field_name": None,
}


class ConnectedComponentsDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        settings: Optional[dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connected components instance segmentation")
        self._layers: list[PointCloudLayer] = []
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.selection_only_box = QCheckBox("On selection only", self)
        form.addRow(self.selection_only_box)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(1e-6, 1e6)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setToolTip("Spatial connectivity radius used by the native CCA.")
        form.addRow("epsilon", self.epsilon_spin)

        self.create_new_radio = QRadioButton("Create new field", self)
        self.write_existing_radio = QRadioButton("Write in existing field", self)
        self.output_group = QButtonGroup(self)
        self.output_group.addButton(self.create_new_radio)
        self.output_group.addButton(self.write_existing_radio)
        form.addRow(self.create_new_radio)

        self.output_field_edit = QLineEdit(self)
        form.addRow("New field base", self.output_field_edit)

        form.addRow(self.write_existing_radio)
        self.existing_field_combo = QComboBox(self)
        form.addRow("Existing instance field", self.existing_field_combo)

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
        self.selection_only_box.toggled.connect(self._validate)
        self.create_new_radio.toggled.connect(self._update_output_state)
        self.write_existing_radio.toggled.connect(self._update_output_state)
        self.output_field_edit.textChanged.connect(self._validate)
        self.existing_field_combo.currentIndexChanged.connect(self._validate)

    def _populate_layers(self, scene: Scene, active_layer: Optional[PointCloudLayer]) -> None:
        self._layers = [
            layer for layer in scene.layers.values() if isinstance(layer, PointCloudLayer)
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
        previous_field = self.existing_field_name()
        self.existing_field_combo.clear()
        layer = self.selected_layer()
        if layer is None:
            self.selection_only_box.setChecked(False)
            self.selection_only_box.setEnabled(False)
            self.write_existing_radio.setEnabled(False)
            if self.write_existing_radio.isChecked():
                self.create_new_radio.setChecked(True)
            self._update_output_state()
            return

        for field in layer.data.get_fields(field_type=FieldType.INSTANCE):
            self.existing_field_combo.addItem(field.name)
        preferred = previous_field
        if layer.active_field is not None and layer.active_field.field_type == FieldType.INSTANCE:
            preferred = layer.active_field.name
        if preferred:
            index = self.existing_field_combo.findText(preferred)
            if index >= 0:
                self.existing_field_combo.setCurrentIndex(index)

        has_instance_fields = self.existing_field_combo.count() > 0
        self.write_existing_radio.setEnabled(has_instance_fields)
        if not has_instance_fields and self.write_existing_radio.isChecked():
            self.create_new_radio.setChecked(True)

        selection = layer.active_selection
        has_selection = selection is not None and selection.size > 0
        self.selection_only_box.setEnabled(has_selection)
        if not has_selection:
            self.selection_only_box.setChecked(False)
        self._update_output_state()

    def _apply_settings(self, settings: dict[str, object]) -> None:
        values = dict(CONNECTED_COMPONENTS_DEFAULTS)
        values.update(settings)
        self.epsilon_spin.setValue(float(values["epsilon"]))
        self.selection_only_box.setChecked(
            bool(values["on_selection_only"]) and self.selection_only_box.isEnabled()
        )
        self.output_field_edit.setText(str(values["output_field_base"]))
        existing_name = values.get("output_existing_field_name")
        if isinstance(existing_name, str):
            index = self.existing_field_combo.findText(existing_name)
            if index >= 0:
                self.existing_field_combo.setCurrentIndex(index)
        if values.get("output_mode") == "write_existing" and self.write_existing_radio.isEnabled():
            self.write_existing_radio.setChecked(True)
        else:
            self.create_new_radio.setChecked(True)

    def _update_output_state(self) -> None:
        create_new = self.output_mode() == "create_new"
        self.output_field_edit.setEnabled(create_new)
        self.existing_field_combo.setEnabled(not create_new and self.write_existing_radio.isEnabled())
        self._validate()

    def _validate(self) -> None:
        valid = self.selected_layer() is not None
        if self.output_mode() == "create_new":
            valid = valid and bool(self.output_field_base())
        else:
            valid = valid and self.existing_field_combo.count() > 0
        if self._ok_button is not None:
            self._ok_button.setEnabled(valid)

    def accept(self) -> None:
        self._validate()
        if self._ok_button is not None and not self._ok_button.isEnabled():
            return
        if self.output_mode() == "write_existing" and not self.on_selection_only():
            field_name = self.existing_field_name()
            if field_name is None or not confirm_global_field_overwrite(self, field_name):
                return
        super().accept()

    def selected_layer(self) -> Optional[PointCloudLayer]:
        layer = self.layer_combo.currentData()
        return layer if isinstance(layer, PointCloudLayer) else None

    def on_selection_only(self) -> bool:
        return self.selection_only_box.isEnabled() and self.selection_only_box.isChecked()

    def output_mode(self) -> str:
        return "write_existing" if self.write_existing_radio.isChecked() else "create_new"

    def output_field_base(self) -> str:
        return self.output_field_edit.text().strip()

    def existing_field_name(self) -> Optional[str]:
        name = self.existing_field_combo.currentText().strip()
        return name or None

    def epsilon(self) -> float:
        return float(self.epsilon_spin.value())

    def settings(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon(),
            "on_selection_only": self.on_selection_only(),
            "output_mode": self.output_mode(),
            "output_field_base": self.output_field_base(),
            "output_existing_field_name": self.existing_field_name(),
        }
