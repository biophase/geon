from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


REGION_MERGE_DEFAULTS: dict[str, object] = {
    "neighbor_radius": 0.05,
    "min_contact_points": 5,
    "planarity_threshold": 0.6,
    "normal_angle_deg": 10.0,
    "plane_distance_threshold": 0.03,
    "min_region_size": 20,
    "output_field_base": "merged_planar_regions",
}


class RegionMergeDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        settings: Optional[dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge planar regions")

        self._layers: list[PointCloudLayer] = []
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.instance_field_combo = QComboBox(self)
        form.addRow("Source instance field", self.instance_field_combo)

        self.neighbor_radius_spin = QDoubleSpinBox(self)
        self.neighbor_radius_spin.setRange(1e-6, 1e6)
        self.neighbor_radius_spin.setDecimals(6)
        self.neighbor_radius_spin.setSingleStep(0.01)
        form.addRow("neighbor radius", self.neighbor_radius_spin)

        self.min_contact_points_spin = QSpinBox(self)
        self.min_contact_points_spin.setRange(1, 1_000_000)
        form.addRow("min contact points", self.min_contact_points_spin)

        self.planarity_threshold_spin = QDoubleSpinBox(self)
        self.planarity_threshold_spin.setRange(0.0, 1.0)
        self.planarity_threshold_spin.setDecimals(3)
        self.planarity_threshold_spin.setSingleStep(0.05)
        form.addRow("planarity threshold", self.planarity_threshold_spin)

        self.normal_angle_spin = QDoubleSpinBox(self)
        self.normal_angle_spin.setRange(0.0, 90.0)
        self.normal_angle_spin.setDecimals(2)
        self.normal_angle_spin.setSingleStep(1.0)
        form.addRow("normal angle (deg)", self.normal_angle_spin)

        self.plane_distance_spin = QDoubleSpinBox(self)
        self.plane_distance_spin.setRange(0.0, 1e6)
        self.plane_distance_spin.setDecimals(6)
        self.plane_distance_spin.setSingleStep(0.01)
        form.addRow("plane distance threshold", self.plane_distance_spin)

        self.min_region_size_spin = QSpinBox(self)
        self.min_region_size_spin.setRange(1, 10_000_000)
        form.addRow("min region size", self.min_region_size_spin)

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
        self._refresh_instance_fields()
        self._apply_settings(settings or {})
        self._validate()

        self.layer_combo.currentIndexChanged.connect(self._refresh_instance_fields)
        self.instance_field_combo.currentIndexChanged.connect(self._validate)
        self.output_field_edit.textChanged.connect(self._validate)

    def _apply_settings(self, settings: dict[str, object]) -> None:
        merged = dict(REGION_MERGE_DEFAULTS)
        merged.update(settings)
        self.neighbor_radius_spin.setValue(float(merged["neighbor_radius"]))
        self.min_contact_points_spin.setValue(int(merged["min_contact_points"]))
        self.planarity_threshold_spin.setValue(float(merged["planarity_threshold"]))
        self.normal_angle_spin.setValue(float(merged["normal_angle_deg"]))
        self.plane_distance_spin.setValue(float(merged["plane_distance_threshold"]))
        self.min_region_size_spin.setValue(int(merged["min_region_size"]))
        self.output_field_edit.setText(str(merged["output_field_base"]))
        source_field_name = merged.get("source_field_name")
        if isinstance(source_field_name, str):
            idx = self.instance_field_combo.findText(source_field_name)
            if idx >= 0:
                self.instance_field_combo.setCurrentIndex(idx)

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

    def _refresh_instance_fields(self) -> None:
        self.instance_field_combo.clear()
        layer = self.selected_layer()
        if layer is None:
            self._validate()
            return
        fields = layer.data.get_fields(field_type=FieldType.INSTANCE)
        for field in fields:
            self.instance_field_combo.addItem(field.name)
        if layer.active_field is not None and layer.active_field.field_type == FieldType.INSTANCE:
            idx = self.instance_field_combo.findText(layer.active_field.name)
            if idx >= 0:
                self.instance_field_combo.setCurrentIndex(idx)
        self._validate()

    def _validate(self) -> None:
        ok = self.selected_layer() is not None
        ok = ok and self.instance_field_combo.count() > 0
        ok = ok and bool(self.output_field_base())
        if self._ok_button is not None:
            self._ok_button.setEnabled(ok)

    def selected_layer(self) -> Optional[PointCloudLayer]:
        layer = self.layer_combo.currentData()
        return layer if isinstance(layer, PointCloudLayer) else None

    def source_field_name(self) -> Optional[str]:
        if self.instance_field_combo.count() == 0:
            return None
        text = self.instance_field_combo.currentText().strip()
        return text if text else None

    def output_field_base(self) -> str:
        return self.output_field_edit.text().strip()

    def params(self) -> dict[str, object]:
        return {
            "neighbor_radius": float(self.neighbor_radius_spin.value()),
            "min_contact_points": int(self.min_contact_points_spin.value()),
            "planarity_threshold": float(self.planarity_threshold_spin.value()),
            "normal_angle_deg": float(self.normal_angle_spin.value()),
            "plane_distance_threshold": float(self.plane_distance_spin.value()),
            "min_region_size": int(self.min_region_size_spin.value()),
        }

    def settings(self) -> dict[str, object]:
        out = self.params()
        out["source_field_name"] = self.source_field_name()
        out["output_field_base"] = self.output_field_base()
        return out
