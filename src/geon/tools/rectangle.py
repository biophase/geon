"""Two-click screen rectangle selection using the lasso projection kernel."""
from dataclasses import dataclass
from typing import ClassVar
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton

from .base import Event
from .lasso import LassoTool
from geon.util.resources import resource_path


@dataclass
class RectangleSelectionTool(LassoTool):
    label: ClassVar = "rectangle_selection"
    tooltip: ClassVar = "Rectangle selection"
    shortcut: ClassVar = "r"
    icon_path: ClassVar = resource_path("rectangle_selection.png")
    cursor_icon_path: ClassVar = resource_path("crosshair.png")
    cursor_hot: ClassVar = (15, 15)
    selection_title: ClassVar = "Rectangle selection"

    def create_context_widget(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.addWidget(QLabel("Click two opposite corners. Escape to cancel.", widget))
        accept = QPushButton("Accept", widget)
        accept.clicked.connect(self._sAccept)
        layout.addWidget(accept)
        cancel = QPushButton("Cancel", widget)
        cancel.clicked.connect(self._sCancel)
        layout.addWidget(cancel)
        return widget

    @staticmethod
    def rectangle(first, second):
        x0, y0 = first
        x1, y1 = second
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _click_event(self, event: Event) -> None:
        if self.select_paused:
            return
        if not self.poly_points:
            self.poly_points = [event.pos]
            return
        first = self.poly_points[0]
        if first[0] == event.pos[0] or first[1] == event.pos[1]:
            return
        self.poly_points = self.rectangle(first, event.pos)
        self.ghost_point = None
        self._sApply("in")
        self._remove_polygon_actor()
        self.select_paused = True
        self.request_deactivate()

    def _mouse_move_event(self, event: Event) -> None:
        if not self.select_paused and self.poly_points:
            first = self.poly_points[0]
            self.poly_points = self.rectangle(first, event.pos)
            self._update_polygon_actor()
            self.poly_points = [first]

    def _sAccept(self) -> None:
        if self.selection_dirty:
            self.request_deactivate()

    def deactivate(self) -> None:
        if not self.selection_dirty:
            self.layer.active_selection = self.selection_backup
        super().deactivate()
