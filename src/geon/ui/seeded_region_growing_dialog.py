from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer


class SeededRegionGrowingDialog(QDialog):
    def __init__(
        self,
        layer: Optional[PointCloudLayer],
        settings: dict[str, object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Live Region Growing Settings")
        self._layer = layer
        self._ok_button = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.normal_mode_combo = QComboBox(self)
        self.normal_mode_combo.addItem("<select normal source>", None)
        self.normal_mode_combo.addItem("Use existing normal field", "use_provided")
        self.normal_mode_combo.addItem("Recompute normals", "compute")
        form.addRow("Normal source", self.normal_mode_combo)

        self.normal_field_combo = QComboBox(self)
        form.addRow("Normals field", self.normal_field_combo)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(1e-6, 1e6)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setSingleStep(0.01)
        form.addRow("epsilon", self.epsilon_spin)

        self.tau_spin = QSpinBox(self)
        self.tau_spin.setRange(3, 10_000_000)
        form.addRow("tau (min points)", self.tau_spin)

        self.alpha_spin = QDoubleSpinBox(self)
        self.alpha_spin.setRange(0.0, 89.9)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setSingleStep(1.0)
        form.addRow("alpha (deg)", self.alpha_spin)

        advanced_group = QGroupBox("Advanced", self)
        advanced_form = QFormLayout(advanced_group)
        layout.addWidget(advanced_group)

        self.enable_seed_gating_box = QCheckBox("Enable seed gating", self)
        self.enable_seed_gating_box.setChecked(True)
        advanced_form.addRow(self.enable_seed_gating_box)

        self.seed_min_neighbors_spin = QSpinBox(self)
        self.seed_min_neighbors_spin.setRange(3, 10_000)
        advanced_form.addRow("Seed min neighbors", self.seed_min_neighbors_spin)

        self.seed_planarity_min_spin = QDoubleSpinBox(self)
        self.seed_planarity_min_spin.setRange(0.0, 1.0)
        self.seed_planarity_min_spin.setDecimals(3)
        self.seed_planarity_min_spin.setSingleStep(0.01)
        advanced_form.addRow("Seed planarity min", self.seed_planarity_min_spin)

        self.seed_scattering_max_spin = QDoubleSpinBox(self)
        self.seed_scattering_max_spin.setRange(0.0, 1.0)
        self.seed_scattering_max_spin.setDecimals(3)
        self.seed_scattering_max_spin.setSingleStep(0.01)
        advanced_form.addRow("Seed scattering max", self.seed_scattering_max_spin)

        self.epsilon_multiplier_spin = QDoubleSpinBox(self)
        self.epsilon_multiplier_spin.setRange(0.1, 100.0)
        self.epsilon_multiplier_spin.setDecimals(3)
        advanced_form.addRow("epsilon multiplier", self.epsilon_multiplier_spin)

        self.refit_multiplier_spin = QDoubleSpinBox(self)
        self.refit_multiplier_spin.setRange(1.1, 100.0)
        self.refit_multiplier_spin.setDecimals(3)
        advanced_form.addRow("Refit multiplier", self.refit_multiplier_spin)

        self.first_refit_spin = QSpinBox(self)
        self.first_refit_spin.setRange(3, 1_000_000)
        advanced_form.addRow("First refit size", self.first_refit_spin)

        self.max_dist_spin = QDoubleSpinBox(self)
        self.max_dist_spin.setRange(0.1, 1_000_000.0)
        self.max_dist_spin.setDecimals(3)
        advanced_form.addRow("Max dist from center", self.max_dist_spin)

        self.oriented_normals_box = QCheckBox("Normals are oriented", self)
        advanced_form.addRow(self.oriented_normals_box)

        self.perform_cca_box = QCheckBox("Connected components cleanup", self)
        self.perform_cca_box.setChecked(True)
        advanced_form.addRow(self.perform_cca_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)

        self._populate_normals()
        self._apply_settings(settings)
        self._update_seed_gate_visibility()
        self._validate()

        self.normal_mode_combo.currentIndexChanged.connect(self._validate)
        self.normal_field_combo.currentIndexChanged.connect(self._validate)
        self.enable_seed_gating_box.toggled.connect(self._update_seed_gate_visibility)

    def _populate_normals(self) -> None:
        self.normal_field_combo.clear()
        if self._layer is None:
            self.normal_field_combo.setEnabled(False)
            return
        normals = [f.name for f in self._layer.data.get_fields(field_type=FieldType.NORMAL)]
        for name in normals:
            self.normal_field_combo.addItem(name)
        self.normal_field_combo.setEnabled(bool(normals))

    def _apply_settings(self, settings: dict[str, object]) -> None:
        normal_mode = settings.get("normal_mode", "compute")
        idx = self.normal_mode_combo.findData(normal_mode)
        self.normal_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

        normal_field_name = settings.get("normal_field_name")
        if isinstance(normal_field_name, str):
            idx = self.normal_field_combo.findText(normal_field_name)
            if idx >= 0:
                self.normal_field_combo.setCurrentIndex(idx)

        self.epsilon_spin.setValue(float(settings.get("epsilon", 0.03)))
        self.tau_spin.setValue(int(settings.get("tau", 80)))
        self.alpha_spin.setValue(float(settings.get("alpha_deg", 29.0)))
        self.enable_seed_gating_box.setChecked(bool(settings.get("enable_seed_gating", True)))
        self.seed_min_neighbors_spin.setValue(int(settings.get("seed_min_neighbors", 10)))
        self.seed_planarity_min_spin.setValue(float(settings.get("seed_planarity_min", 0.20)))
        self.seed_scattering_max_spin.setValue(float(settings.get("seed_scattering_max", 0.35)))
        self.epsilon_multiplier_spin.setValue(float(settings.get("epsilon_multiplier", 3.0)))
        self.refit_multiplier_spin.setValue(float(settings.get("refit_multiplier", 2.0)))
        self.first_refit_spin.setValue(int(settings.get("first_refit", 4)))
        self.max_dist_spin.setValue(float(settings.get("max_dist_from_cent", 50.0)))
        self.oriented_normals_box.setChecked(bool(settings.get("oriented_normals", False)))
        self.perform_cca_box.setChecked(bool(settings.get("perform_cca", True)))

    def _update_seed_gate_visibility(self) -> None:
        enabled = self.enable_seed_gating_box.isChecked()
        self.seed_min_neighbors_spin.setEnabled(enabled)
        self.seed_planarity_min_spin.setEnabled(enabled)
        self.seed_scattering_max_spin.setEnabled(enabled)

    def _validate(self) -> None:
        ok = True
        use_provided = self.normal_mode() == "use_provided"
        self.normal_field_combo.setEnabled(use_provided and self.normal_field_combo.count() > 0)
        if self.normal_mode() is None:
            ok = False
        if use_provided and self.normal_field_name() is None:
            ok = False
        if self._ok_button is not None:
            self._ok_button.setEnabled(ok)

    def normal_mode(self) -> Optional[str]:
        mode = self.normal_mode_combo.currentData()
        return mode if isinstance(mode, str) else None

    def normal_field_name(self) -> Optional[str]:
        if self.normal_field_combo.count() == 0:
            return None
        text = self.normal_field_combo.currentText().strip()
        return text if text else None

    def settings(self) -> dict[str, object]:
        return {
            "normal_mode": self.normal_mode() or "compute",
            "normal_field_name": self.normal_field_name(),
            "epsilon": float(self.epsilon_spin.value()),
            "tau": int(self.tau_spin.value()),
            "alpha_deg": float(self.alpha_spin.value()),
            "enable_seed_gating": bool(self.enable_seed_gating_box.isChecked()),
            "seed_min_neighbors": int(self.seed_min_neighbors_spin.value()),
            "seed_planarity_min": float(self.seed_planarity_min_spin.value()),
            "seed_scattering_max": float(self.seed_scattering_max_spin.value()),
            "epsilon_multiplier": float(self.epsilon_multiplier_spin.value()),
            "refit_multiplier": float(self.refit_multiplier_spin.value()),
            "first_refit": int(self.first_refit_spin.value()),
            "max_dist_from_cent": float(self.max_dist_spin.value()),
            "oriented_normals": bool(self.oriented_normals_box.isChecked()),
            "perform_cca": bool(self.perform_cca_box.isChecked()),
        }
