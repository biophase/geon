from PyQt6.QtWidgets import (QWidget, QDockWidget, QLabel, QToolButton, QHBoxLayout, QTreeWidget,
                             QVBoxLayout, QGridLayout,QPushButton)

from PyQt6.QtCore import Qt, QSize








class VTKViewer(QWidget):
    """
    VTK Widget
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("VTK VIEW PLACEHOLDER", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "QLabel { "
            "   background: #333333; "
            "   color: #DDDDDD; "
            "   font-size: 18px; "
            "   font-weight: bold; "
            "}"
        )
        layout.addWidget(label)

