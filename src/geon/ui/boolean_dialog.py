from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt

from ..core.constants import Boolean
from typing import Optional

class BooleanChoiceDialog(QDialog):
    def __init__(self, parent=None, message:Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Boolean Operation")
        self.setModal(True)
        self.default: Boolean = Boolean.OVERWRITE
        self.choice: Optional[Boolean] = None

        layout = QVBoxLayout(self)
        self._button_group = QButtonGroup(self)

        if message is not None:
            label = QLabel()
            label.setText(message)
            layout.addWidget(label)
        for bool_value in Boolean:
            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            radio = QRadioButton(row)
            icon_path = Boolean.icon_path(bool_value)
            radio.setIcon(QIcon(QPixmap(icon_path)))
            radio.setText(bool_value.name.lower().capitalize())
            radio.setChecked(bool_value == self.default)
            self._button_group.addButton(radio)
            self._button_group.setId(radio, int(bool_value.value))

            row_layout.addWidget(radio)
            row_layout.addStretch()
            layout.addWidget(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self._on_reject)
        layout.addWidget(buttons)

    def showEvent(self, a0) -> None:
        event = a0
        super().showEvent(event)
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self.grabKeyboard()

    def closeEvent(self, a0) -> None:
        event=a0
        self.releaseKeyboard()
        super().closeEvent(event)

    def _on_accept(self) -> None:
        button = self._button_group.checkedButton()
        if button is None:
            self.choice = self.default
        else:
            button_id = self._button_group.id(button)
            self.choice = Boolean(button_id)
        self.releaseKeyboard()
        self.accept()

    def _on_reject(self) -> None:
        self.choice = None
        self.releaseKeyboard()
        self.reject()
