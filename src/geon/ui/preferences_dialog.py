from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
)

from geon.settings import Preferences


class PreferencesDialog(QDialog):
    def __init__(self, preferences: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._prefs = preferences

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.user_input = QLineEdit(self)
        self.user_input.setText(self._prefs.user_name)
        form.addRow("User name", self.user_input)

        self.telemetry_checkbox = QCheckBox(self)
        self.telemetry_checkbox.setChecked(self._prefs.enable_telemetry)
        form.addRow("Enable telemetry", self.telemetry_checkbox)

        self.camera_sensitivity_input = QDoubleSpinBox(self)
        self.camera_sensitivity_input.setDecimals(3)
        self.camera_sensitivity_input.setRange(0.01, 1000.0)
        self.camera_sensitivity_input.setSingleStep(0.5)
        self.camera_sensitivity_input.setValue(float(self._prefs.camera_sensitivity))
        form.addRow("Camera sensitivity", self.camera_sensitivity_input)

        self.cell_size_mode_combo = QComboBox(self)
        self.cell_size_mode_combo.addItem("Screen space", "screen")
        self.cell_size_mode_combo.addItem("World space", "world")
        mode_idx = self.cell_size_mode_combo.findData(self._prefs.cell_complex_size_mode)
        self.cell_size_mode_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        form.addRow("Cell cube size mode", self.cell_size_mode_combo)

        self.cell_screen_size_input = QDoubleSpinBox(self)
        self.cell_screen_size_input.setDecimals(1)
        self.cell_screen_size_input.setRange(1.0, 200.0)
        self.cell_screen_size_input.setSingleStep(1.0)
        self.cell_screen_size_input.setValue(float(self._prefs.cell_complex_screen_size_px))
        form.addRow("Cell cube screen size (px)", self.cell_screen_size_input)

        self.cell_world_size_input = QDoubleSpinBox(self)
        self.cell_world_size_input.setDecimals(4)
        self.cell_world_size_input.setRange(0.0001, 1_000_000.0)
        self.cell_world_size_input.setSingleStep(0.1)
        self.cell_world_size_input.setValue(float(self._prefs.cell_complex_world_size))
        form.addRow("Cell cube world size", self.cell_world_size_input)

        self.cell_edge_width_input = QDoubleSpinBox(self)
        self.cell_edge_width_input.setDecimals(1)
        self.cell_edge_width_input.setRange(1.0, 20.0)
        self.cell_edge_width_input.setSingleStep(1.0)
        self.cell_edge_width_input.setValue(float(self._prefs.cell_complex_edge_width))
        form.addRow("Cell edge width", self.cell_edge_width_input)

        self.cell_reference_label_text_size_input = QDoubleSpinBox(self)
        self.cell_reference_label_text_size_input.setDecimals(1)
        self.cell_reference_label_text_size_input.setRange(6.0, 72.0)
        self.cell_reference_label_text_size_input.setSingleStep(1.0)
        self.cell_reference_label_text_size_input.setValue(
            float(self._prefs.cell_complex_reference_label_text_size_px)
        )
        form.addRow("Reference label text size (px)", self.cell_reference_label_text_size_input)

        color_row = QLineEdit(self)
        color_row.setText(",".join(str(int(c)) for c in self._prefs.cell_complex_default_color))
        form.addRow("Cell default RGB", color_row)
        self.cell_color_text = color_row

        selection_color_row = QLineEdit(self)
        selection_color_row.setText(",".join(str(int(c)) for c in self._prefs.selection_color))
        form.addRow("Selection RGB", selection_color_row)
        self.selection_color_text = selection_color_row

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self) -> None:
        self._prefs.user_name = self.user_input.text().strip() or "Unnamed User"
        self._prefs.enable_telemetry = self.telemetry_checkbox.isChecked()
        self._prefs.camera_sensitivity = float(self.camera_sensitivity_input.value())
        mode = self.cell_size_mode_combo.currentData()
        self._prefs.cell_complex_size_mode = str(mode or "screen")
        self._prefs.cell_complex_screen_size_px = float(self.cell_screen_size_input.value())
        self._prefs.cell_complex_world_size = float(self.cell_world_size_input.value())
        self._prefs.cell_complex_edge_width = float(self.cell_edge_width_input.value())
        self._prefs.cell_complex_reference_label_text_size_px = float(
            self.cell_reference_label_text_size_input.value()
        )
        try:
            rgb = [int(part.strip()) for part in self.cell_color_text.text().split(",")]
            if len(rgb) != 3:
                raise ValueError
            self._prefs.cell_complex_default_color = [
                int(max(0, min(255, c))) for c in rgb
            ]
        except ValueError:
            self._prefs.cell_complex_default_color = [204, 204, 204]
        try:
            rgb = [int(part.strip()) for part in self.selection_color_text.text().split(",")]
            if len(rgb) != 3:
                raise ValueError
            self._prefs.selection_color = [
                int(max(0, min(255, c))) for c in rgb
            ]
        except ValueError:
            self._prefs.selection_color = [255, 128, 0]
