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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.pointcloud import FieldType
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


class RegionGrowingDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Planar region growing")

        self._layers: list[PointCloudLayer] = []
        self._normal_fields_by_layer: dict[int, list[str]] = {}
        self._ok_button: Optional[QPushButton] = None

        layout = QVBoxLayout(self)

        form = QFormLayout()
        layout.addLayout(form)

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(1e-6, 1e6)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.setValue(0.03)
        form.addRow("epsilon", self.epsilon_spin)

        self.tau_spin = QSpinBox(self)
        self.tau_spin.setRange(3, 10_000_000)
        self.tau_spin.setValue(80)
        form.addRow("tau (min points)", self.tau_spin)

        self.alpha_spin = QDoubleSpinBox(self)
        self.alpha_spin.setRange(0.0, 89.9)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setSingleStep(1.0)
        self.alpha_spin.setValue(29.0)
        form.addRow("alpha (deg)", self.alpha_spin)

        estimate_row = QWidget(self)
        estimate_layout = QHBoxLayout(estimate_row)
        estimate_layout.setContentsMargins(0, 0, 0, 0)
        self.estimate_sample_spin = QSpinBox(self)
        self.estimate_sample_spin.setRange(100, 5_000_000)
        self.estimate_sample_spin.setValue(50_000)
        self.estimate_sample_spin.setToolTip("Sample size for fast parameter estimation")
        estimate_layout.addWidget(QLabel("Sample", estimate_row))
        estimate_layout.addWidget(self.estimate_sample_spin)
        self.estimate_seed_spin = QSpinBox(self)
        self.estimate_seed_spin.setRange(0, 2**31 - 1)
        self.estimate_seed_spin.setValue(0)
        estimate_layout.addWidget(QLabel("Seed", estimate_row))
        estimate_layout.addWidget(self.estimate_seed_spin)
        self.estimate_button = QPushButton("Estimate", estimate_row)
        estimate_layout.addWidget(self.estimate_button)
        self.fast_preset_button = QPushButton("Fast preset", estimate_row)
        estimate_layout.addWidget(self.fast_preset_button)
        form.addRow("Estimator", estimate_row)

        self.normal_mode_combo = QComboBox(self)
        self.normal_mode_combo.addItem("<select normal source>", None)
        self.normal_mode_combo.addItem("Use existing normal field", "use_provided")
        self.normal_mode_combo.addItem("Recompute normals", "compute")
        form.addRow("Normal source", self.normal_mode_combo)

        self.normal_field_combo = QComboBox(self)
        form.addRow("Normals field", self.normal_field_combo)

        self.field_name_edit = QLineEdit(self)
        self.field_name_edit.setText("planar_regions")
        form.addRow("Output field base", self.field_name_edit)

        advanced_group = QGroupBox("Advanced", self)
        advanced_form = QFormLayout(advanced_group)
        layout.addWidget(advanced_group)

        self.confidence_spin = QDoubleSpinBox(self)
        self.confidence_spin.setRange(0.5, 0.999999)
        self.confidence_spin.setDecimals(6)
        self.confidence_spin.setSingleStep(0.001)
        self.confidence_spin.setValue(0.99)
        advanced_form.addRow("Stop confidence", self.confidence_spin)

        self.enable_seed_gating_box = QCheckBox("Enable seed gating", self)
        self.enable_seed_gating_box.setChecked(True)
        self.enable_seed_gating_box.setToolTip(
            "Filter candidate seeds by local planarity/scattering before growth starts."
        )
        advanced_form.addRow(self.enable_seed_gating_box)

        self.seed_min_neighbors_spin = QSpinBox(self)
        self.seed_min_neighbors_spin.setRange(3, 10_000)
        self.seed_min_neighbors_spin.setValue(10)
        self.seed_min_neighbors_spin.setToolTip(
            "Minimum epsilon-neighborhood size required for a seed to be considered planar."
        )
        advanced_form.addRow("Seed min neighbors", self.seed_min_neighbors_spin)

        self.seed_planarity_min_spin = QDoubleSpinBox(self)
        self.seed_planarity_min_spin.setRange(0.0, 1.0)
        self.seed_planarity_min_spin.setDecimals(3)
        self.seed_planarity_min_spin.setSingleStep(0.01)
        self.seed_planarity_min_spin.setValue(0.20)
        self.seed_planarity_min_spin.setToolTip(
            "Seed gate: minimum local planarity (l2-l1)/l3."
        )
        advanced_form.addRow("Seed planarity min", self.seed_planarity_min_spin)

        self.seed_scattering_max_spin = QDoubleSpinBox(self)
        self.seed_scattering_max_spin.setRange(0.0, 1.0)
        self.seed_scattering_max_spin.setDecimals(3)
        self.seed_scattering_max_spin.setSingleStep(0.01)
        self.seed_scattering_max_spin.setValue(0.35)
        self.seed_scattering_max_spin.setToolTip(
            "Seed gate: maximum local scattering l1/l3."
        )
        advanced_form.addRow("Seed scattering max", self.seed_scattering_max_spin)

        self.failrate_window_spin = QSpinBox(self)
        self.failrate_window_spin.setRange(8, 10_000)
        self.failrate_window_spin.setValue(64)
        self.failrate_window_spin.setToolTip(
            "Rolling window size (attempts) for fail-rate stopping."
        )
        advanced_form.addRow("Fail-rate window", self.failrate_window_spin)

        self.failrate_threshold_spin = QDoubleSpinBox(self)
        self.failrate_threshold_spin.setRange(0.5, 0.999)
        self.failrate_threshold_spin.setDecimals(3)
        self.failrate_threshold_spin.setSingleStep(0.01)
        self.failrate_threshold_spin.setValue(0.90)
        self.failrate_threshold_spin.setToolTip(
            "Stop chunk when rolling fail-rate exceeds this threshold."
        )
        advanced_form.addRow("Fail-rate threshold", self.failrate_threshold_spin)

        self.chunk_mode_combo = QComboBox(self)
        self.chunk_mode_combo.addItem("Auto target points", "auto")
        self.chunk_mode_combo.addItem("Explicit xyz", "explicit")
        advanced_form.addRow("Chunk mode", self.chunk_mode_combo)

        self.enable_chunking_box = QCheckBox("Enable chunking", self)
        self.enable_chunking_box.setChecked(True)
        advanced_form.addRow(self.enable_chunking_box)

        self.target_points_spin = QSpinBox(self)
        self.target_points_spin.setRange(1000, 50_000_000)
        self.target_points_spin.setValue(250_000)
        advanced_form.addRow("Target points/chunk", self.target_points_spin)

        explicit_chunk_row = QWidget(self)
        explicit_chunk_layout = QHBoxLayout(explicit_chunk_row)
        explicit_chunk_layout.setContentsMargins(0, 0, 0, 0)
        self.chunk_x_spin = QSpinBox(self)
        self.chunk_x_spin.setRange(1, 256)
        self.chunk_x_spin.setValue(2)
        self.chunk_y_spin = QSpinBox(self)
        self.chunk_y_spin.setRange(1, 256)
        self.chunk_y_spin.setValue(2)
        self.chunk_z_spin = QSpinBox(self)
        self.chunk_z_spin.setRange(1, 256)
        self.chunk_z_spin.setValue(1)
        explicit_chunk_layout.addWidget(QLabel("X", explicit_chunk_row))
        explicit_chunk_layout.addWidget(self.chunk_x_spin)
        explicit_chunk_layout.addWidget(QLabel("Y", explicit_chunk_row))
        explicit_chunk_layout.addWidget(self.chunk_y_spin)
        explicit_chunk_layout.addWidget(QLabel("Z", explicit_chunk_row))
        explicit_chunk_layout.addWidget(self.chunk_z_spin)
        advanced_form.addRow("Chunk xyz", explicit_chunk_row)

        self.overlap_factor_spin = QDoubleSpinBox(self)
        self.overlap_factor_spin.setRange(0.0, 20.0)
        self.overlap_factor_spin.setDecimals(3)
        self.overlap_factor_spin.setValue(3.0)
        advanced_form.addRow("Overlap factor", self.overlap_factor_spin)

        self.merge_angle_spin = QDoubleSpinBox(self)
        self.merge_angle_spin.setRange(0.0, 90.0)
        self.merge_angle_spin.setDecimals(2)
        self.merge_angle_spin.setValue(5.0)
        advanced_form.addRow("Merge angle (deg)", self.merge_angle_spin)

        self.merge_dist_factor_spin = QDoubleSpinBox(self)
        self.merge_dist_factor_spin.setRange(0.0, 20.0)
        self.merge_dist_factor_spin.setDecimals(3)
        self.merge_dist_factor_spin.setValue(3.0)
        advanced_form.addRow("Merge distance factor", self.merge_dist_factor_spin)

        self.enable_reconciliation_box = QCheckBox("Enable reconciliation", self)
        self.enable_reconciliation_box.setChecked(True)
        advanced_form.addRow(self.enable_reconciliation_box)

        self.epsilon_multiplier_spin = QDoubleSpinBox(self)
        self.epsilon_multiplier_spin.setRange(0.1, 100.0)
        self.epsilon_multiplier_spin.setDecimals(3)
        self.epsilon_multiplier_spin.setValue(3.0)
        advanced_form.addRow("epsilon multiplier", self.epsilon_multiplier_spin)

        self.refit_multiplier_spin = QDoubleSpinBox(self)
        self.refit_multiplier_spin.setRange(1.1, 100.0)
        self.refit_multiplier_spin.setDecimals(3)
        self.refit_multiplier_spin.setValue(2.0)
        advanced_form.addRow("Refit multiplier", self.refit_multiplier_spin)

        self.first_refit_spin = QSpinBox(self)
        self.first_refit_spin.setRange(3, 1_000_000)
        self.first_refit_spin.setValue(4)
        advanced_form.addRow("First refit size", self.first_refit_spin)

        self.max_dist_spin = QDoubleSpinBox(self)
        self.max_dist_spin.setRange(0.1, 1_000_000.0)
        self.max_dist_spin.setDecimals(3)
        self.max_dist_spin.setValue(50.0)
        advanced_form.addRow("Max dist from center", self.max_dist_spin)

        self.oriented_normals_box = QCheckBox("Normals are oriented", self)
        advanced_form.addRow(self.oriented_normals_box)
        self.perform_cca_box = QCheckBox("Connected components cleanup", self)
        self.perform_cca_box.setChecked(True)
        advanced_form.addRow(self.perform_cca_box)

        self.refine_unassigned_box = QCheckBox("Reassign unassigned leftovers", self)
        self.refine_unassigned_box.setChecked(True)
        advanced_form.addRow(self.refine_unassigned_box)

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
        self._update_chunk_mode_visibility()
        self._validate()

        self.layer_combo.currentIndexChanged.connect(self._refresh_normals_fields)
        self.normal_mode_combo.currentIndexChanged.connect(self._validate)
        self.normal_field_combo.currentIndexChanged.connect(self._validate)
        self.field_name_edit.textChanged.connect(self._validate)
        self.chunk_mode_combo.currentIndexChanged.connect(self._update_chunk_mode_visibility)
        self.chunk_mode_combo.currentIndexChanged.connect(self._validate)
        self.enable_chunking_box.toggled.connect(self._update_chunk_mode_visibility)
        self.enable_seed_gating_box.toggled.connect(self._update_seed_gating_visibility)
        self.fast_preset_button.clicked.connect(self.apply_fast_preset)
        self._update_seed_gating_visibility()

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

    def _update_chunk_mode_visibility(self) -> None:
        if not self.enable_chunking_box.isChecked():
            self.chunk_mode_combo.setEnabled(False)
            self.target_points_spin.setEnabled(False)
            self.chunk_x_spin.setEnabled(False)
            self.chunk_y_spin.setEnabled(False)
            self.chunk_z_spin.setEnabled(False)
            return
        self.chunk_mode_combo.setEnabled(True)
        mode = self.chunk_mode()
        is_auto = (mode == "auto")
        self.target_points_spin.setEnabled(is_auto)
        self.chunk_x_spin.setEnabled(not is_auto)
        self.chunk_y_spin.setEnabled(not is_auto)
        self.chunk_z_spin.setEnabled(not is_auto)

    def _update_seed_gating_visibility(self) -> None:
        enabled = self.enable_seed_gating_box.isChecked()
        self.seed_min_neighbors_spin.setEnabled(enabled)
        self.seed_planarity_min_spin.setEnabled(enabled)
        self.seed_scattering_max_spin.setEnabled(enabled)

    def apply_fast_preset(self) -> None:
        # Conservative speed-oriented defaults for large clouds.
        self.confidence_spin.setValue(0.95)
        self.enable_seed_gating_box.setChecked(True)
        self.seed_min_neighbors_spin.setValue(12)
        self.seed_planarity_min_spin.setValue(0.25)
        self.seed_scattering_max_spin.setValue(0.30)
        self.failrate_window_spin.setValue(48)
        self.failrate_threshold_spin.setValue(0.85)
        self.enable_chunking_box.setChecked(True)
        self.chunk_mode_combo.setCurrentIndex(self.chunk_mode_combo.findData("auto"))
        self.target_points_spin.setValue(1_500_000)
        self.overlap_factor_spin.setValue(1.5)
        self.enable_reconciliation_box.setChecked(False)
        self.perform_cca_box.setChecked(False)
        self.refine_unassigned_box.setChecked(False)
        self._update_chunk_mode_visibility()

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

    def epsilon(self) -> float:
        return float(self.epsilon_spin.value())

    def tau(self) -> int:
        return int(self.tau_spin.value())

    def alpha_deg(self) -> float:
        return float(self.alpha_spin.value())

    def set_estimated_core(self, epsilon: float, tau: int, alpha_deg: float) -> None:
        self.epsilon_spin.setValue(float(epsilon))
        self.tau_spin.setValue(int(max(3, tau)))
        self.alpha_spin.setValue(float(max(0.0, min(89.9, alpha_deg))))

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

    def estimate_kwargs(self) -> dict[str, int]:
        return {
            "sample_size": int(self.estimate_sample_spin.value()),
            "seed": int(self.estimate_seed_spin.value()),
        }

    def params(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon(),
            "tau": self.tau(),
            "alpha_deg": self.alpha_deg(),
            "confidence": float(self.confidence_spin.value()),
            "epsilon_multiplier": float(self.epsilon_multiplier_spin.value()),
            "refit_multiplier": float(self.refit_multiplier_spin.value()),
            "first_refit": int(self.first_refit_spin.value()),
            "max_dist_from_cent": float(self.max_dist_spin.value()),
            "oriented_normals": bool(self.oriented_normals_box.isChecked()),
            "perform_cca": bool(self.perform_cca_box.isChecked()),
            "refine_unassigned": bool(self.refine_unassigned_box.isChecked()),
            "enable_seed_gating": bool(self.enable_seed_gating_box.isChecked()),
            "seed_min_neighbors": int(self.seed_min_neighbors_spin.value()),
            "seed_planarity_min": float(self.seed_planarity_min_spin.value()),
            "seed_scattering_max": float(self.seed_scattering_max_spin.value()),
            "failrate_window": int(self.failrate_window_spin.value()),
            "failrate_threshold": float(self.failrate_threshold_spin.value()),
        }

    def chunk_mode(self) -> str:
        mode = self.chunk_mode_combo.currentData()
        return mode if isinstance(mode, str) else "auto"

    def chunking(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enable_chunking_box.isChecked()),
            "mode": self.chunk_mode(),
            "target_points_per_chunk": int(self.target_points_spin.value()),
            "chunk_x": int(self.chunk_x_spin.value()),
            "chunk_y": int(self.chunk_y_spin.value()),
            "chunk_z": int(self.chunk_z_spin.value()),
            "overlap_factor": float(self.overlap_factor_spin.value()),
        }

    def merge(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enable_reconciliation_box.isChecked()),
            "angle_deg": float(self.merge_angle_spin.value()),
            "distance_factor": float(self.merge_dist_factor_spin.value()),
        }
