from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from geon.data.cellcomplex import Cell, PointCloudRef
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.util.resources import resource_path


def _format_value(value: Any) -> str:
    return str(value)


class GeometryRefPropertiesDialog(QDialog):
    def __init__(self, properties: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference properties")
        layout = QVBoxLayout(self)
        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setRowCount(len(properties))
        table.setHorizontalHeaderLabels(["Property", "Value"])
        for row, (key, value) in enumerate(properties.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(_format_value(value)))
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CellComplexAttributesDialog(QDialog):
    def __init__(self, layer: CellComplexLayer, cell: Cell, ctx: Any, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell attributes")
        self._layer = layer
        self._cell = cell
        self._ctx = ctx

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        tabs.addTab(self._make_references_tab(), "References")
        tabs.addTab(self._make_attributes_tab(), "Attributes")
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _layer_for_ref(self, ref_id: str):
        scene = getattr(self._ctx, "scene", None)
        if scene is None:
            return None
        return scene.layers.get(ref_id)

    def _make_references_tab(self) -> QTableWidget:
        table = QTableWidget(self)
        refs = list(self._cell.geometry_refs)
        table.setColumnCount(4)
        table.setRowCount(len(refs))
        table.setHorizontalHeaderLabels(["", "Layer ID", "View", "Clear"])
        for row, ref in enumerate(refs):
            layer = self._layer_for_ref(ref.ref_id)
            icon_item = QTableWidgetItem("")
            if isinstance(layer, PointCloudLayer):
                icon_item.setIcon(QIcon(resource_path("tree_icon_pointcloud.png")))
            table.setItem(row, 0, icon_item)
            table.setItem(row, 1, QTableWidgetItem(ref.ref_id))

            view_btn = QPushButton("View", table)
            view_btn.clicked.connect(
                lambda _checked=False, r=ref: GeometryRefPropertiesDialog(
                    r.get_properties(),
                    self,
                ).exec()
            )
            table.setCellWidget(row, 2, view_btn)

            clear_btn = QPushButton("Clear", table)
            clear_btn.clicked.connect(lambda _checked=False, r=ref: self._clear_ref(r))
            table.setCellWidget(row, 3, clear_btn)
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        return table

    def _make_attributes_tab(self) -> QTableWidget:
        table = QTableWidget(self)
        items = list(self._cell.attributes.items())
        table.setColumnCount(2)
        table.setRowCount(len(items))
        table.setHorizontalHeaderLabels(["Attribute", "Value"])
        for row, (key, value) in enumerate(items):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(_format_value(value)))
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        return table

    def _clear_ref(self, ref: PointCloudRef) -> None:
        self._cell.geometry_refs = [r for r in self._cell.geometry_refs if r is not ref]
        mark_modified = getattr(self._ctx.viewer.window(), "_mark_active_doc_modified", None)
        if callable(mark_modified):
            mark_modified()
        self.accept()
        CellComplexAttributesDialog(self._layer, self._cell, self._ctx, self.parent()).exec()
