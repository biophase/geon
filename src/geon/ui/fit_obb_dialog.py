from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..data.pointcloud import FieldType
from ..rendering.boundingbox import BoundingBoxLayer
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


class FitObbDialog(QDialog):
    def __init__(
        self,
        scene: Scene,
        source_layer: PointCloudLayer,
        selection_count: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit oriented bounding box")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        form.addRow("Point cloud", QLabel(source_layer.browser_name, self))
        form.addRow("Working points", QLabel(f"{selection_count:,}", self))

        self.target_layer_combo = QComboBox(self)
        for layer in scene.layers.values():
            if isinstance(layer, BoundingBoxLayer):
                self.target_layer_combo.addItem(layer.browser_name, layer)
        form.addRow("Bounding-box layer", self.target_layer_combo)

        self.method_combo = QComboBox(self)
        self.method_combo.addItem("PCA", "pca")
        self.method_combo.addItem("PCA, Z locked", "pca_z_locked")
        form.addRow("Method", self.method_combo)

        self.separate_box = QCheckBox("Separate with field", self)
        form.addRow(self.separate_box)
        self.separate_field_combo = QComboBox(self)
        for field in source_layer.data.get_fields():
            if field.field_type in (FieldType.INSTANCE, FieldType.SEMANTIC):
                self.separate_field_combo.addItem(field.name)
        self.separate_field_combo.setEnabled(False)
        if self.separate_field_combo.count() == 0:
            self.separate_box.setEnabled(False)
        form.addRow("Separation field", self.separate_field_combo)
        self.separate_box.toggled.connect(self.separate_field_combo.setEnabled)

        self.trim_spin = QDoubleSpinBox(self)
        self.trim_spin.setRange(0.0, 0.499)
        self.trim_spin.setDecimals(4)
        self.trim_spin.setSingleStep(0.005)
        self.trim_spin.setValue(0.01)
        self.trim_spin.setToolTip("Fraction trimmed independently from each end of every projected axis.")
        form.addRow("Trim per side", self.trim_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.target_layer_combo.count() > 0 and selection_count > 0
        )

    def target_layer(self) -> BoundingBoxLayer | None:
        layer = self.target_layer_combo.currentData()
        return layer if isinstance(layer, BoundingBoxLayer) else None

    def method(self) -> str:
        return str(self.method_combo.currentData())

    def trim(self) -> float:
        return float(self.trim_spin.value())

    def separate_with_field(self) -> bool:
        return self.separate_box.isEnabled() and self.separate_box.isChecked()

    def separation_field_name(self) -> str | None:
        if not self.separate_with_field():
            return None
        name = self.separate_field_combo.currentText().strip()
        return name or None
