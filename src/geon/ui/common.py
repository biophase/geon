from geon.io.dataset import Dataset
from geon.data.document import Document


from typing import Optional
import os.path as osp

from PyQt6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QDockWidget, QWidget, 
                             QStackedWidget, QTreeWidgetItem, QFileDialog, QVBoxLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from PyQt6.QtGui import QFontMetrics

class DockTitleBar(QWidget):
    def __init__(self, dock: QDockWidget):
        super().__init__(dock)
        self.dock = dock
        label = QLabel(dock.windowTitle())
        label_font = label.font()
        label_font.setBold(True)
        label.setFont(label_font)
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

class Dock(QDockWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "_"))
        self.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
            |   QDockWidget.DockWidgetFeature.DockWidgetFloatable
            |   QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setTitleBarWidget(DockTitleBar(self))


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle):
        super().__init__("", parent)
        self._full_text = text
        self._elide_mode = mode
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self._update_elided_text()

    def setText(self, a0: Optional[str]) -> None:
        self._full_text = a0
        self._update_elided_text()

    def fullText(self) -> Optional[str]:
        return self._full_text

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        fm = QFontMetrics(self.font())
        width = max(self.width(), 0)
        elided = fm.elidedText(self._full_text, self._elide_mode, width)
        super().setText(elided)
        
