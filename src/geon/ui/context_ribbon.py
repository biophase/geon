from PyQt6.QtWidgets import (QToolBar, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QToolButton, QPushButton)
from PyQt6.QtCore import QSize, Qt

class ContextRibbon(QToolBar):
    def __init__(self, parent=None):
        super().__init__("Context Ribbon", parent)

        # ribbon config
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(24,24))

        # Example layout idea: # TODO: remove dummy impelmentation
        # [ Selection group ] [ Transform group ] [ Display group ]

        self._add_selection_group()
        self.addSeparator()
        self._add_transform_group()
        self.addSeparator()
        # self._add_display_group()

    def _make_group(self, title: str, buttons: list[str], highlight=False) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(4, 2, 4, 2)

        title_label = QLabel(title)
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 10pt; }")

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(2)

        for text in buttons:
            btn = QPushButton()
            btn.setText(text)
            # btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(QSize(72, 56))
            btn.setFlat(True)

            button_row.addWidget(btn)

        outer.addWidget(title_label)
        outer.addLayout(button_row)

        # ⭐ Highlight group if requested
        if highlight:
            w.setStyleSheet("""
                QWidget {
                    background-color: rgba(0, 255, 0, 60);   /* semi-transparent green */
                    border-radius: 1px;
                }
            """)

        return w


    def _add_selection_group(self):
        group = self._make_group(
            "Selection",
            ["Select", "Lasso"],
        )
        self.addWidget(group)

    def _add_transform_group(self):
        group = self._make_group(
            "Transform",
            ["Move", "Rotate", "Scale"], highlight=True)
        self.addWidget(group)

    def _add_display_group(self):
        group = self._make_group(
            "Display",
            ["Wireframe", "Shaded"],
        )
        self.addWidget(group)
