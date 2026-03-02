from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QHBoxLayout,
    QWidget,
)

from ..rendering.scene import Scene
from ..rendering.pointcloud import PointCloudLayer


class FeaturesDialog(QDialog):
    _OPTIONAL_FEATURES: list[tuple[str, str]] = [
        ("sum_eigenvalues", "Sum of eigenvalues"),
        ("omnivariance", "Omnivariance"),
        ("eigenentropy", "Eigenentropy"),
        ("anisotropy", "Anisotropy"),
        ("planarity", "Planarity"),
        ("linearity", "Linearity"),
        ("pca1", "PCA1"),
        ("pca2", "PCA2"),
        ("surface_variation", "Surface variation"),
        ("sphericity", "Sphericity"),
        ("verticality", "Verticality"),
        ("eigenvalue_1", "1st eigenvalue"),
        ("eigenvalue_2", "2nd eigenvalue"),
        ("eigenvalue_3", "3rd eigenvalue"),
    ]

    def __init__(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute geometric features")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self._layers: list[PointCloudLayer] = []

        self.layer_combo = QComboBox(self)
        form.addRow("Point cloud layer", self.layer_combo)

        self.radius_spin = QDoubleSpinBox(self)
        self.radius_spin.setRange(1e-6, 1e6)
        self.radius_spin.setDecimals(4)
        self.radius_spin.setSingleStep(0.01)
        self.radius_spin.setValue(0.1)
        form.addRow("Radius", self.radius_spin)

        self.compute_normals_box, self.normals_name_edit = self._add_feature_row(
            form,
            "Compute normals",
            checked=True,
        )
        self.compute_eigenvals_box, self.eigenvals_name_edit = self._add_feature_row(
            form,
            "Compute eigenvalues",
            checked=True,
        )

        self._optional_feature_rows: dict[str, tuple[QCheckBox, QLineEdit]] = {}
        for key, label in self._OPTIONAL_FEATURES:
            box, edit = self._add_feature_row(form, label, checked=False)
            self._optional_feature_rows[key] = (box, edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._populate_layers(scene, active_layer)

    def _add_feature_row(
        self,
        form: QFormLayout,
        label: str,
        *,
        checked: bool,
    ) -> tuple[QCheckBox, QLineEdit]:
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox(label, row_widget)
        checkbox.setChecked(checked)
        row_layout.addWidget(checkbox)

        name_edit = QLineEdit(row_widget)
        name_edit.setPlaceholderText("<default>")
        name_edit.setEnabled(checked)
        row_layout.addWidget(name_edit, stretch=1)

        checkbox.toggled.connect(name_edit.setEnabled)
        form.addRow(row_widget)
        return checkbox, name_edit

    def _populate_layers(
        self,
        scene: Scene,
        active_layer: Optional[PointCloudLayer],
    ) -> None:
        self._layers = [
            layer for layer in scene.layers.values()
            if isinstance(layer, PointCloudLayer)
        ]
        if not self._layers:
            self.layer_combo.addItem("<no point clouds>")
            self.layer_combo.setEnabled(False)
            if self._ok_button is not None:
                self._ok_button.setEnabled(False)
            return

        for layer in self._layers:
            self.layer_combo.addItem(layer.browser_name, layer)

        if active_layer is not None and active_layer in self._layers:
            self.layer_combo.setCurrentIndex(self._layers.index(active_layer))

    @staticmethod
    def _text_or_none(edit: QLineEdit) -> Optional[str]:
        text = edit.text().strip()
        return text if text else None

    def selected_layer(self) -> Optional[PointCloudLayer]:
        data = self.layer_combo.currentData()
        return data if isinstance(data, PointCloudLayer) else None

    def radius(self) -> float:
        return float(self.radius_spin.value())

    def compute_normals(self) -> bool:
        return self.compute_normals_box.isChecked()

    def compute_eigenvals(self) -> bool:
        return self.compute_eigenvals_box.isChecked()

    def normals_field_name(self) -> Optional[str]:
        if not self.compute_normals():
            return None
        return self._text_or_none(self.normals_name_edit)

    def eigenvals_field_name(self) -> Optional[str]:
        if not self.compute_eigenvals():
            return None
        return self._text_or_none(self.eigenvals_name_edit)

    def optional_feature_field_names(self) -> dict[str, Optional[str]]:
        selected: dict[str, Optional[str]] = {}
        for key, (box, edit) in self._optional_feature_rows.items():
            if box.isChecked():
                selected[key] = self._text_or_none(edit)
        return selected


