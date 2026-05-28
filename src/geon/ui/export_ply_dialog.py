from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from geon.data.pointcloud import PointCloudData
from geon.io.ply import PLY_DTYPE_CHOICES


def _dtype_label(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    for label, candidate in PLY_DTYPE_CHOICES.items():
        if dtype == candidate:
            return label
    if np.issubdtype(dtype, np.floating):
        return "float32"
    if np.issubdtype(dtype, np.unsignedinteger):
        return "uint32"
    if np.issubdtype(dtype, np.integer):
        return "int32"
    return "float32"


@dataclass
class PlyExportOptions:
    path: str
    field_names: list[str]
    field_dtypes: dict[str, str]
    coord_dtype: str


class PlyExportDialog(QDialog):
    def __init__(self, pcd: PointCloudData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Point Cloud to PLY")
        self._pcd = pcd
        self._field_rows: dict[str, tuple[QCheckBox, QComboBox]] = {}

        layout = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        browse_btn = QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(QLabel("File:", self))
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        form = QFormLayout()
        self.coord_dtype_combo = self._make_dtype_combo(_dtype_label(self._pcd.points.dtype))
        form.addRow("Coordinate dtype", self.coord_dtype_combo)
        layout.addLayout(form)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Export", "Field", "Type", "Shape", "PLY dtype"])
        layout.addWidget(self.table)
        self._populate_fields()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_dtype_combo(self, selected: str) -> QComboBox:
        combo = QComboBox(self)
        for label in PLY_DTYPE_CHOICES:
            combo.addItem(label, label)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else combo.findData("float32"))
        return combo

    def _populate_fields(self) -> None:
        fields = self._pcd.get_fields()
        self.table.setRowCount(len(fields))
        for row, field in enumerate(fields):
            checkbox = QCheckBox(self.table)
            checkbox.setChecked(True)
            checkbox.setToolTip("Export this field")
            self.table.setCellWidget(row, 0, checkbox)

            name_item = QTableWidgetItem(field.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, name_item)

            type_item = QTableWidgetItem(field.field_type.human_name)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, type_item)

            shape_item = QTableWidgetItem("x".join(str(v) for v in field.data.shape))
            shape_item.setFlags(shape_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, shape_item)

            combo = self._make_dtype_combo(_dtype_label(field.data.dtype))
            self.table.setCellWidget(row, 4, combo)
            self._field_rows[field.name] = (checkbox, combo)

        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Point Cloud to PLY",
            "",
            "PLY files (*.ply);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        self.path_edit.setText(path)

    def options(self) -> PlyExportOptions:
        field_names: list[str] = []
        field_dtypes: dict[str, str] = {}
        for name, (checkbox, combo) in self._field_rows.items():
            if not checkbox.isChecked():
                continue
            field_names.append(name)
            field_dtypes[name] = str(combo.currentData())
        return PlyExportOptions(
            path=self.path_edit.text().strip(),
            field_names=field_names,
            field_dtypes=field_dtypes,
            coord_dtype=str(self.coord_dtype_combo.currentData()),
        )
