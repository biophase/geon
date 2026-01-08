from geon.tools.base import Event
from .base import ModeTool, ToolZone
from ..rendering.pointcloud import PointCloudLayer
from ..data.pointcloud import FieldType, FieldBase, SemanticSegmentation
from geon.util.resources import resource_path

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QBrush

import numpy as np


def _format_val(val: object) -> str:
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            val = val.item()
        else:
            return np.array2string(val, precision=4, separator=", ")
    # handle Python floats and numpy floating scalars
    if isinstance(val, (float, np.floating)):
        return f"{val:.4g}"
    return str(val)


@dataclass
class InspectTool(ModeTool):
    # general settings
    label: ClassVar = 'inspect'
    tooltip: ClassVar = "Inspect tool"
    icon_path: ClassVar = resource_path('inspect_tool.png')
    shortcut: ClassVar = None
    ui_zones: ClassVar = {ToolZone.SIDEBAR_RIGHT_ESSENTIALS}
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = True
    cursor_icon_path : ClassVar = resource_path('inspect_tool.png')
    cursor_hot: ClassVar = (3, 3) 
    
    # mode tool settings
    keep_focus: ClassVar[bool] = False

    _popup: Optional[QDialog] = field(default=None, init=False, repr=False)
    _expanded_fields: set[str] = field(default_factory=set, init=False, repr=False)
    _last_idx: Optional[int] = field(default=None, init=False, repr=False)
    
    def _close_popup(self) -> None:
        if self._popup is not None:
            self._popup.close()
            self._popup = None

    def _add_rows_for_field(self, rows: list[tuple[str, str, object]], field: FieldBase, value: np.ndarray) -> None:
        ft = field.field_type
        if ft == FieldType.SEMANTIC:
            flat = np.asarray(value).reshape(-1)
            val_int = int(flat[0]) if flat.size else -1
            if isinstance(field, SemanticSegmentation) and field.schema is not None:
                try:
                    sem_cls = field.schema.by_id(val_int)
                    color = sem_cls.color
                    label = f"{sem_cls.name} (ID={sem_cls.id})"
                except Exception:
                    color = (180, 180, 180)
                    label = f"ID={val_int}"
            else:
                color = (180, 180, 180)
                label = f"ID={val_int}"
            rows.append((field.name, "", ("semantic", color, label)))
            return
        if value.ndim == 0:
            rows.append((field.name, "", value.item()))
            return
        if value.ndim == 1:
            comps = value
        else:
            comps = value.reshape(-1)
        if ft == FieldType.COLOR:
            labels = ["Red", "Blue", "Green"]
            for lbl, val in zip(labels, comps):
                rows.append((field.name, lbl, val))
        elif ft == FieldType.NORMAL:
            labels = ["nX", "nY", "nZ"]
            for lbl, val in zip(labels, comps[:3]):
                rows.append((field.name, lbl, val))
        elif ft == FieldType.VECTOR:
            field_key = field.name
            expanded = field_key in self._expanded_fields
            to_show = comps if expanded else comps[:3]
            for idx, val in enumerate(to_show):
                rows.append((field.name if idx == 0 else "", str(idx), val))
            if not expanded and comps.shape[0] > 3:
                rows.append((field.name, "", ("expand", comps.shape[0], field_key)))
        else:
            for idx, val in enumerate(comps):
                rows.append((field.name if idx == 0 else "", str(idx), val))

    def _build_rows(self, layer: PointCloudLayer, picked_idx: int) -> list[tuple[str, str, object]]:
        rows: list[tuple[str, str, object]] = []
        coords = layer.data.points[picked_idx]
        for lbl, val in zip(["X", "Y", "Z"], coords[:3]):
            rows.append(("Coord.", lbl, val))
        for field in layer.data.get_fields():
            data = field.data
            if picked_idx >= data.shape[0]:
                continue
            value = np.asarray(data[picked_idx])
            self._add_rows_for_field(rows, field, value)
        return rows

    def _populate_table(self, table: QTableWidget, rows: list[tuple[str, str, object]]) -> None:
        table.setRowCount(len(rows))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Field", "Index", "Value"])
        header = table.horizontalHeader()
        if header is None:
            return
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for r, (fname, idx_label, val) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(fname))
            table.setItem(r, 1, QTableWidgetItem(idx_label))
            if isinstance(val, tuple) and val and val[0] == "expand":
                _, total_dims, field_key = val
                btn = QPushButton(f"expand {total_dims} rows", table)
                btn.clicked.connect(lambda _=None, fk=field_key: self._on_expand(fk))
                table.setCellWidget(r, 2, btn)
            elif isinstance(val, tuple) and val and val[0] == "semantic":
                _, color, label = val
                item = QTableWidgetItem(label)
                qcolor = QColor(*color)
                item.setBackground(QBrush(qcolor))
                # optional readability: set text color based on luminance
                luminance = 0.2126 * qcolor.red() + 0.7152 * qcolor.green() + 0.0722 * qcolor.blue()
                text_color = QColor(0, 0, 0) if luminance > 128 else QColor(255, 255, 255)
                item.setForeground(QBrush(text_color))
                table.setItem(r, 2, item)
            else:
                table.setItem(r, 2, QTableWidgetItem(_format_val(val)))

    def _on_expand(self, field_key: str) -> None:
        self._expanded_fields.add(field_key)
        self._refresh_popup()

    def _refresh_popup(self) -> None:
        if self._popup is None or not isinstance(self.ctx.scene.active_layer, PointCloudLayer):
            return
        picked_idx = getattr(self, "_last_idx", None)
        if picked_idx is None:
            return
        layer = self.ctx.scene.active_layer
        rows = self._build_rows(layer, picked_idx)
        table = self._popup.findChild(QTableWidget)
        if table is not None:
            self._populate_table(table, rows)

    def _show_popup(self, layer: PointCloudLayer, idx: int, click_pos: Optional[tuple[int, int]]) -> None:
        self._close_popup()
        self._expanded_fields.clear()
        self._last_idx = idx
        dlg = QDialog(self.ctx.viewer)
        dlg.setWindowTitle("Point inspection")
        dlg.setWindowFlag(Qt.WindowType.Tool)
        dlg.setModal(False)
        layout = QVBoxLayout(dlg)
        table = QTableWidget(dlg)
        table.setMaximumHeight(320)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._populate_table(table, self._build_rows(layer, idx))
        layout.addWidget(table)
        dlg.show()
        if click_pos is not None:
            x, y = click_pos
            y_flipped = self.ctx.viewer.height() - y
            global_pos = self.ctx.viewer.mapToGlobal(QPoint(x, y_flipped))
            dlg.move(global_pos)
        self._popup = dlg

    def _on_click(self, event: Event) -> None:
        result = self.ctx.viewer.pick()
        if result.layer is None:
            return
        self._close_popup()
        if result.layer.id == self.ctx.scene.active_layer_id:
            if isinstance(result.layer, PointCloudLayer):
                idx = result.element_idx
                if idx is None:
                    return
                self._show_popup(result.layer, idx, event.pos)
            else:
                raise NotImplementedError(f"No inspect implementation for {type(result.layer)}")

    # ------------------------------------------------
    # hooks
    # ------------------------------------------------
    
    def left_button_press_hook(self, event: Event) -> None:
        self._on_click(event)
        super().left_button_press_hook(event)
        
    def activate(self) -> None:
        return super().activate()
    
    def deactivate(self) -> None:
        self._close_popup()
        return super().deactivate()

    def key_press_hook(self, event: Event) -> None:
        super().key_press_hook(event)
        if event.key is None:
            return
        if event.key.lower() == 'escape':
            self.ctx.controller.deactivate_tool()
