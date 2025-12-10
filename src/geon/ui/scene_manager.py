from .common import Dock, ElidedLabel

from geon.rendering.scene import Scene

from PyQt6.QtWidgets import (QStackedWidget, QLabel, QWidget, QVBoxLayout, QTreeWidget)
from PyQt6.QtCore import Qt

from typing import Optional


class SceneManager(Dock):
    def __init__(self, parent=None):
        super().__init__("Scene", parent)
        self._scene : Optional[Scene] =  None
        
        # the UI stacks two cases: 1) no scene loaded and 2) scene loaded
        self.stack = QStackedWidget()
        self.overlay_label = QLabel("No Dataset Work Directory set")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("font-size: 16px; color: gray;")
        
        page = QWidget()
        self.tree_layout = QVBoxLayout(page)
        self.scene_label = ElidedLabel("")
        
        self.tree = QTreeWidget(self)
        self.tree_layout.addWidget(self.scene_label)
        self.tree_layout.addWidget(self.tree)
        
        self.stack.addWidget(self.overlay_label)    # index 0
        self.stack.addWidget(page)                  # index 1
        
        