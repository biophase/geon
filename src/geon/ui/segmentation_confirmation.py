"""Shared confirmation for segmentation operations that overwrite a field."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QWidget


def confirm_global_field_overwrite(parent: QWidget, field_name: str) -> bool:
    warning = QMessageBox(parent)
    warning.setWindowTitle("Overwrite segmentation field")
    warning.setIcon(QMessageBox.Icon.Warning)
    warning.setTextFormat(Qt.TextFormat.PlainText)
    warning.setText(
        f"You are about to overwrite the field `{field_name}` globally. "
        "Are you sure you want to do this? This can't be undone."
    )
    warning.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    warning.setDefaultButton(QMessageBox.StandardButton.No)
    warning.setEscapeButton(QMessageBox.StandardButton.No)
    return warning.exec() == QMessageBox.StandardButton.Yes
