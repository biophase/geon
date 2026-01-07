from PyQt6.QtWidgets import (
    QToolBar, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QToolButton, 
    QPushButton, QLayout
    
    )
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction

from typing import Optional, Literal
from config.theme import UIStyle


class ContextRibbon(QToolBar):
    RIBBON_HEIGHT = 80
    def __init__(self, parent=None):
        super().__init__("Context Ribbon", parent)

        # ribbon config
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(24,24))
        self.setFixedHeight(self.RIBBON_HEIGHT)
        
        self._group_widgets: dict[str, Optional[QWidget]] = {
            "layer": None,
            "selection": None,
            "tool": None,
        }
        self._group_actions: dict[str, Optional[QAction]] = {
            "layer": None,
            "selection": None,
            "tool": None,
        }
        self._sep_actions: list[QAction] = []
        # self._add_selection_group()
        # self.addSeparator()
        
        # self._add_transform_group()
        # self.addSeparator()
        
        # self._add_display_group()

    def set_group(self,
                  title: str,
                  group_content: QWidget | None,
                  group_type: Literal['layer', 'selection', 'tool']
                  ) -> None:
        self.clear_group(group_type)
        if group_content is None:
            return
        w = self._make_group(title, group_content, group_type)
        self._group_widgets[group_type] = w
        self._insert_group_action(group_type, w)
        self._refresh_separators()

    def clear_group(self, group_type: Literal['layer', 'selection', 'tool']) -> None:
        existing = self._group_widgets.get(group_type)
        action = self._group_actions.get(group_type)
        if action is not None:
            self.removeAction(action)
            self._group_actions[group_type] = None
        if existing is None:
            return
        self._group_widgets[group_type] = None
        existing.deleteLater()
        self._refresh_separators()

    def _make_group(self, 
                    title: str, 
                    group_content: QWidget, 
                    group_type: Literal['layer', 'selection', 'tool']
                    ) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(4, 2, 4, 2)

        title_label = QLabel(title)
        # title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(2)
        content_row.addWidget(group_content)

        outer.addWidget(title_label)
        outer.addLayout(content_row)

        if group_type == "tool":
            w.setObjectName("tool_group")
            w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            w.setStyleSheet(
                "QWidget#tool_group {"
                "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                "stop:0 rgba(255, 127, 0, 127), stop:1 rgba(255, 127, 0, 30));"
                "border-radius: 4px;"
                "border: 1px solid rgba(255, 127, 0, 180);"
                "}"
            )
        else:
            w.setObjectName("group")
            w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            w.setStyleSheet(
                "QWidget#group {"
                # "background: transparent;" 
                "background: solid rgba(128,128,128,30);"
                "border: 1px solid rgba(128, 128, 128, 50);"
                "border-radius: 4px;"
                
                "}"
            )

        return w
        
    def _update(self) -> None:
        ...

    def _insert_group_action(self, group_type: str, widget: QWidget) -> None:
        ordered = ["layer", "selection", "tool"]
        idx = ordered.index(group_type)
        before_action = None
        for next_type in ordered[idx + 1:]:
            next_action = self._group_actions.get(next_type)
            if next_action is not None:
                before_action = next_action
                break
        if before_action is None:
            action = self.addWidget(widget)
        else:
            action = self.insertWidget(before_action, widget)
        self._group_actions[group_type] = action

    def _refresh_separators(self) -> None:
        for sep in self._sep_actions:
            self.removeAction(sep)
        self._sep_actions.clear()

        boundaries = [("layer", "selection"), ("selection", "tool")]
        for left, right in boundaries:
            left_action = self._group_actions.get(left)
            right_action = self._group_actions.get(right)
            if left_action is None or right_action is None:
                continue
            sep = self.insertSeparator(right_action)
            if sep is not None:
                self._sep_actions.append(sep)
