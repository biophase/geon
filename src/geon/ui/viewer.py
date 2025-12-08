from PyQt6.QtWidgets import (QWidget, QDockWidget, QLabel, QToolButton, QHBoxLayout, QTreeWidget,
                             QVBoxLayout, QGridLayout,QPushButton)

from PyQt6.QtCore import Qt, QSize




class DockTitleBar(QWidget):
    def __init__(self, dock: QDockWidget):
        super().__init__(dock)
        self.dock = dock
        label = QLabel(dock.windowTitle())
        detach_btn = QPushButton()
        detach_btn.setText("⧉")
        detach_btn.setToolTip("Detach window")
        detach_btn.setFlat(True)
        detach_btn.setFixedSize(QSize(24,24))
        detach_btn.clicked.connect(lambda: dock.setFloating(True))
        
        layout= QHBoxLayout(self)
        layout.setContentsMargins(4,0,4,0)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(detach_btn)

class TreeDock(QDockWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "_"))
        tree = QTreeWidget()
        tree.setHeaderLabels(["Name"])
        self.setWidget(tree)
        self.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
            |   QDockWidget.DockWidgetFeature.DockWidgetFloatable
            |   QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setTitleBarWidget(DockTitleBar(self))


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

