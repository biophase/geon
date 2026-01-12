from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QCheckBox,
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
