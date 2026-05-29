from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QFormLayout,
    QVBoxLayout,
)

from geon.data.cellcomplex import Cell, PointCloudRef
from geon.data.pointcloud import SemanticSchema
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.util.resources import resource_path
from geon.ui.semantic_schema_dialog import SemanticSchemaCreationDialog


def _format_value(value: Any) -> str:
    return str(value)


class CreateCellSemanticAttributeDialog(QDialog):
    CREATE_SCHEMA_SENTINEL = "__create_schema__"

    def __init__(
        self,
        schemas: list[SemanticSchema],
        taken_schema_names: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create semantic attribute")
        self._schemas = list(schemas)
        self._taken_schema_names = list(taken_schema_names)
        self.attribute_name: Optional[str] = None
        self.dimension: Optional[int] = None
        self.schema: Optional[SemanticSchema] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit(self)
        self.name_edit.textChanged.connect(lambda _text: self._update_state())
        form.addRow("Attribute name:", self.name_edit)

        self.name_warning = QLabel("Names cannot contain spaces.", self)
        self.name_warning.setStyleSheet("QLabel { color: #b00020; }")
        self.name_warning.setVisible(False)
        form.addRow("", self.name_warning)

        self.schema_combo = QComboBox(self)
        self.schema_combo.currentIndexChanged.connect(self._on_schema_changed)
        form.addRow("Schema:", self.schema_combo)

        self.dim_combo = QComboBox(self)
        self.dim_combo.addItem("Node", 0)
        self.dim_combo.addItem("Edge", 1)
        self.dim_combo.currentIndexChanged.connect(lambda _idx: self._update_state())
        form.addRow("Dimension:", self.dim_combo)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        accept_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if accept_btn is not None:
            accept_btn.setText("Accept")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._populate_schema_combo()
        self._update_state()

    def _populate_schema_combo(self, selected_schema: Optional[SemanticSchema] = None) -> None:
        self.schema_combo.blockSignals(True)
        self.schema_combo.clear()
        selected_index = -1
        for schema in self._schemas:
            self.schema_combo.addItem(schema.name, schema)
            if selected_schema is schema:
                selected_index = self.schema_combo.count() - 1
        self.schema_combo.addItem("<create new schema>", self.CREATE_SCHEMA_SENTINEL)
        if selected_index >= 0:
            self.schema_combo.setCurrentIndex(selected_index)
        self.schema_combo.blockSignals(False)

    def _on_schema_changed(self, index: int) -> None:
        if self.schema_combo.itemData(index) != self.CREATE_SCHEMA_SENTINEL:
            self._update_state()
            return
        dlg = SemanticSchemaCreationDialog(
            required_ids=[],
            taken_schema_names=self._taken_schema_names,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.schema is not None:
            self._schemas.append(dlg.schema)
            self._taken_schema_names.append(dlg.schema.name)
            self._populate_schema_combo(dlg.schema)
        elif self.schema_combo.count() > 1:
            self.schema_combo.setCurrentIndex(0)
        self._update_state()

    def _selected_schema(self) -> Optional[SemanticSchema]:
        data = self.schema_combo.currentData()
        return data if isinstance(data, SemanticSchema) else None

    def _name_error(self) -> Optional[str]:
        name = self.name_edit.text().strip()
        if not name:
            return "Name is required."
        if any(ch.isspace() for ch in name):
            return "Names cannot contain spaces."
        return None

    def _update_state(self) -> None:
        name_error = self._name_error()
        self.name_warning.setText(name_error or "")
        self.name_warning.setVisible(name_error is not None and bool(self.name_edit.text()))
        self.name_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #b00020; }"
            if self.name_warning.isVisible()
            else ""
        )
        accept_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if accept_btn is not None:
            accept_btn.setEnabled(
                name_error is None
                and self.dim_combo.currentData() is not None
                and self._selected_schema() is not None
            )

    def accept(self) -> None:
        if self._name_error() is not None or self._selected_schema() is None:
            self._update_state()
            return
        self.attribute_name = self.name_edit.text().strip()
        self.dimension = int(self.dim_combo.currentData())
        self.schema = self._selected_schema()
        super().accept()


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
        tabs.addTab(self._make_semantic_attributes_tab(), "Semantic attributes")
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

    def _make_semantic_attributes_tab(self) -> QTableWidget:
        table = QTableWidget(self)
        items = list(self._cell.semantic_attributes.items())
        table.setColumnCount(4)
        table.setRowCount(len(items))
        table.setHorizontalHeaderLabels(["Attribute", "Class", "ID", "Schema"])
        schemas = self._layer.data.semantic_attribute_schemas.get(self._cell.dim, {})
        for row, (key, class_id) in enumerate(items):
            schema = schemas.get(key)
            class_name = ""
            schema_name = ""
            if schema is not None:
                schema_name = schema.name
                try:
                    class_name = schema.by_id(int(class_id)).name
                except IndexError:
                    class_name = "<invalid>"
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(class_name))
            table.setItem(row, 2, QTableWidgetItem(str(class_id)))
            table.setItem(row, 3, QTableWidgetItem(schema_name))
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        return table

    def _clear_ref(self, ref: PointCloudRef) -> None:
        self._cell.geometry_refs = [r for r in self._cell.geometry_refs if r is not ref]
        mark_modified = getattr(self._ctx.viewer.window(), "_mark_active_doc_modified", None)
        if callable(mark_modified):
            mark_modified()
        self.accept()
        CellComplexAttributesDialog(self._layer, self._cell, self._ctx, self.parent()).exec()
