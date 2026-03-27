from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


PLANE_RANSAC_DEFAULTS: dict[str, object] = {
    "epsilon": 0.03,
    "min_points": 100,
    "normal_threshold_deg": 25.0,
    "cluster_epsilon": -1.0,
    "probability": 0.01,
    "normal_mode": "compute",
    "normal_field_name": None,
    "output_field_base": "ransac_planes",
    "max_iterations_per_plane": 5000,
    "seed": 0,
}


class PlaneRansacDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        settings: Optional[dict[str, object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plane RANSAC")

        self._layers: list[PointCloudLayer] = []
        self._normal_fields_by_layer: dict[int, list[str]] = {}
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(1e-6, 1e6)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setSingleStep(0.01)
        form.addRow("epsilon", self.epsilon_spin)

        self.min_points_spin = QSpinBox(self)
        self.min_points_spin.setRange(3, 10_000_000)
        form.addRow("min points", self.min_points_spin)

        self.normal_threshold_spin = QDoubleSpinBox(self)
        self.normal_threshold_spin.setRange(0.0, 89.9)
        self.normal_threshold_spin.setDecimals(2)
        self.normal_threshold_spin.setSingleStep(1.0)
        form.addRow("normal threshold (deg)", self.normal_threshold_spin)

        self.cluster_epsilon_spin = QDoubleSpinBox(self)
        self.cluster_epsilon_spin.setRange(0.0, 1e6)
        self.cluster_epsilon_spin.setDecimals(6)
        self.cluster_epsilon_spin.setSingleStep(0.01)
        self.cluster_epsilon_spin.setSpecialValueText("Use epsilon")
        form.addRow("cluster epsilon", self.cluster_epsilon_spin)

        self.probability_spin = QDoubleSpinBox(self)
        self.probability_spin.setRange(1e-6, 0.999999)
        self.probability_spin.setDecimals(6)
        self.probability_spin.setSingleStep(0.01)
        form.addRow("probability", self.probability_spin)

        self.normal_mode_combo = QComboBox(self)
        self.normal_mode_combo.addItem("<select normal source>", None)
        self.normal_mode_combo.addItem("Use existing normal field", "use_provided")
        self.normal_mode_combo.addItem("Recompute normals", "compute")
        form.addRow("Normal source", self.normal_mode_combo)

        self.normal_field_combo = QComboBox(self)
        form.addRow("Normals field", self.normal_field_combo)

        self.field_name_edit = QLineEdit(self)
        form.addRow("Output field base", self.field_name_edit)

        advanced_group = QGroupBox("Advanced", self)
        advanced_form = QFormLayout(advanced_group)
        layout.addWidget(advanced_group)

        self.max_iterations_spin = QSpinBox(self)
        self.max_iterations_spin.setRange(100, 1_000_000)
        advanced_form.addRow("Max iterations/plane", self.max_iterations_spin)

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 2**31 - 1)
        advanced_form.addRow("Seed", self.seed_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        self._populate_layers(scene, active_layer)
        self._refresh_normals_fields()
        self._apply_settings(settings or {})
        self._validate()

        self.layer_combo.currentIndexChanged.connect(self._refresh_normals_fields)
        self.normal_mode_combo.currentIndexChanged.connect(self._validate)
        self.normal_field_combo.currentIndexChanged.connect(self._validate)
        self.field_name_edit.textChanged.connect(self._validate)

    def _apply_settings(self, settings: dict[str, object]) -> None:
        merged = dict(PLANE_RANSAC_DEFAULTS)
        merged.update(settings)
        self.epsilon_spin.setValue(float(merged["epsilon"]))
        self.min_points_spin.setValue(int(merged["min_points"]))
        self.normal_threshold_spin.setValue(float(merged["normal_threshold_deg"]))
        cluster_eps = float(merged["cluster_epsilon"])
        self.cluster_epsilon_spin.setValue(max(0.0, cluster_eps))
        self.probability_spin.setValue(float(merged["probability"]))
        self.field_name_edit.setText(str(merged["output_field_base"]))
        self.max_iterations_spin.setValue(int(merged["max_iterations_per_plane"]))
        self.seed_spin.setValue(int(merged["seed"]))

        normal_mode = merged.get("normal_mode")
        mode_idx = self.normal_mode_combo.findData(normal_mode)
        self.normal_mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)

        normal_field_name = merged.get("normal_field_name")
        if isinstance(normal_field_name, str):
            field_idx = self.normal_field_combo.findText(normal_field_name)
            if field_idx >= 0:
                self.normal_field_combo.setCurrentIndex(field_idx)

    def _populate_layers(self, scene: Scene, active_layer: Optional[PointCloudLayer]) -> None:
        self._layers = [
            layer for layer in scene.layers.values()
            if isinstance(layer, PointCloudLayer)
        ]
        self.layer_combo.clear()
        self._normal_fields_by_layer.clear()

        if not self._layers:
            self.layer_combo.addItem("<no point clouds>")
            self.layer_combo.setEnabled(False)
            return

        for layer in self._layers:
            self.layer_combo.addItem(layer.browser_name, layer)
            normals = [f.name for f in layer.data.get_fields(field_type=FieldType.NORMAL)]
            self._normal_fields_by_layer[id(layer)] = normals

        if active_layer is not None and active_layer in self._layers:
            self.layer_combo.setCurrentIndex(self._layers.index(active_layer))

    def _refresh_normals_fields(self) -> None:
        self.normal_field_combo.clear()
        layer = self.selected_layer()
        if layer is None:
            self.normal_field_combo.setEnabled(False)
            self._validate()
            return
        normals = self._normal_fields_by_layer.get(id(layer), [])
        for name in normals:
            self.normal_field_combo.addItem(name)
        self.normal_field_combo.setEnabled(len(normals) > 0)
        self._validate()

    def _validate(self) -> None:
        ok = True
        use_provided = self.normal_mode() == "use_provided"
        self.normal_field_combo.setEnabled(use_provided and self.normal_field_combo.count() > 0)
        if self.selected_layer() is None:
            ok = False
        if self.normal_mode() is None:
            ok = False
        if use_provided and self.normal_field_name() is None:
            ok = False
        if not self.output_field_base().strip():
            ok = False
        if self._ok_button is not None:
            self._ok_button.setEnabled(ok)

    def selected_layer(self) -> Optional[PointCloudLayer]:
        layer = self.layer_combo.currentData()
        return layer if isinstance(layer, PointCloudLayer) else None

    def normal_mode(self) -> Optional[str]:
        mode = self.normal_mode_combo.currentData()
        return mode if isinstance(mode, str) else None

    def normal_field_name(self) -> Optional[str]:
        if self.normal_field_combo.count() == 0:
            return None
        text = self.normal_field_combo.currentText().strip()
        return text if text else None

    def output_field_base(self) -> str:
        return self.field_name_edit.text().strip()

    def params(self) -> dict[str, object]:
        cluster_epsilon = float(self.cluster_epsilon_spin.value())
        return {
            "epsilon": float(self.epsilon_spin.value()),
            "min_points": int(self.min_points_spin.value()),
            "normal_threshold_deg": float(self.normal_threshold_spin.value()),
            "cluster_epsilon": cluster_epsilon if cluster_epsilon > 0.0 else -1.0,
            "probability": float(self.probability_spin.value()),
            "max_iterations_per_plane": int(self.max_iterations_spin.value()),
            "seed": int(self.seed_spin.value()),
        }

    def settings(self) -> dict[str, object]:
        out = self.params()
        out["normal_mode"] = self.normal_mode()
        out["normal_field_name"] = self.normal_field_name()
        out["output_field_base"] = self.output_field_base()
        return out
