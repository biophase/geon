from __future__ import annotations

from typing import Dict, Any, Optional, List, cast

import numpy as np
from plyfile import PlyData

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QWidget,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDialogButtonBox,
    QMessageBox,
)

# ---------------------------------------------------------------------------
# Adjust these imports to match your package structure
# ---------------------------------------------------------------------------
# from .point_cloud_data_module import PointCloudData, FieldType, SemanticSchema, ColorMap
from geon.data.pointcloud import (
    PointCloudData,
    FieldType,
    SemanticSchema,
)
from geon.data.definitions import ColorMap  # adjust if different


class ImportPLYDialog(QDialog):
    """
    Dialog that maps a .ply point cloud to your PointCloudData structure.

    Parameters
    ----------
    ply_path : str
        Path to the .ply file to import.
    semantic_schemas : Dict[str, SemanticSchema]
        Mapping of schema name -> SemanticSchema.
        (Currently used by picking the first available schema for semantic fields.)
    color_maps : Dict[str, ColorMap]
        Mapping of colormap name -> ColorMap.
        (Currently used by picking the first available colormap for color fields.)
    """

    # Input table columns
    IN_STATUS_COL = 0
    IN_NAME_COL = 1
    IN_DTYPE_COL = 2
    IN_MAP_FIELD_COL = 3
    IN_MAP_INDEX_COL = 4

    # Output table columns
    OUT_NAME_COL = 0
    OUT_TYPE_COL = 1
    OUT_NCOLS_COL = 2
    OUT_BTN_COL = 3

    def __init__(
        self,
        ply_path: str,
        semantic_schemas: Dict[str, SemanticSchema],
        color_maps: Dict[str, ColorMap],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.ply_path = ply_path
        self.semantic_schemas = semantic_schemas
        self.color_maps = color_maps

        self.setWindowTitle("Import PLY")
        self.resize(1100, 650)

        self._suppress_output_item_changed = False
        self._point_cloud: Optional[PointCloudData] = None

        self._load_ply()
        self._build_ui()
        self._populate_input_fields()
        self._setup_default_output_fields()
        self._refresh_input_mapping_targets()

    # ------------------------------------------------------------------
    # Accessor for the resulting PointCloudData
    # ------------------------------------------------------------------

    @property
    def point_cloud(self) -> Optional[PointCloudData]:
        return self._point_cloud

    # ------------------------------------------------------------------
    # PLY loading
    # ------------------------------------------------------------------

    def _load_ply(self) -> None:
        """Load the PLY file and store the vertex element."""
        self.ply_data = PlyData.read(self.ply_path)

        if "vertex" not in self.ply_data:
            QMessageBox.critical(
                self,
                "PLY error",
                "PLY file has no 'vertex' element. "
                "Adapt the dialog if you need to support other elements.",
            )
            self.vertex_element = None
            return

        self.vertex_element = self.ply_data["vertex"]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left: Input fields (PLY)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_label = QLabel("Input fields (PLY)")
        left_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(left_label)

        self.input_table = QTableWidget(0, 5, self)
        self.input_table.setHorizontalHeaderLabels(
            ["", "Field name", "Dtype", "Mapped to", "Index"]
        )
        cast(QHeaderView, self.input_table.horizontalHeader()).setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.input_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.input_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        left_layout.addWidget(self.input_table)

        # Right: Output fields (PointCloudData)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_label = QLabel("Point cloud data")
        right_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(right_label)

        self.output_table = QTableWidget(0, 4, self)
        self.output_table.setHorizontalHeaderLabels(
            ["Name", "Field type", "# columns", ""]
        )
        cast(QHeaderView, self.output_table.horizontalHeader()).setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.output_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.output_table.itemChanged.connect(self._on_output_item_changed)
        right_layout.addWidget(self.output_table)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Populate input table from PLY
    # ------------------------------------------------------------------

    def _populate_input_fields(self) -> None:
        if self.vertex_element is None:
            return

        self.input_table.setRowCount(0)
        dtype = self.vertex_element.data.dtype
        names = dtype.names or []

        for name in names:
            field_dtype = dtype.fields[name][0]
            self._add_input_row(name, str(field_dtype))

    def _add_input_row(self, name: str, dtype_str: str) -> None:
        row = self.input_table.rowCount()
        self.input_table.insertRow(row)

        # Column 0: status ('?' or '✓')
        status_item = QTableWidgetItem("?")
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        status_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.input_table.setItem(row, self.IN_STATUS_COL, status_item)

        # Column 1: field name
        name_item = QTableWidgetItem(name)
        name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.input_table.setItem(row, self.IN_NAME_COL, name_item)

        # Column 2: dtype
        dtype_item = QTableWidgetItem(dtype_str)
        dtype_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.input_table.setItem(row, self.IN_DTYPE_COL, dtype_item)

        # Column 3: mapped-to field (combo box)
        map_combo = QComboBox(self.input_table)
        map_combo.addItem("")  # empty = not mapped
        map_combo.currentTextChanged.connect(
            lambda _txt, r=row: self._on_input_mapping_changed(r)
        )
        self.input_table.setCellWidget(row, self.IN_MAP_FIELD_COL, map_combo)

        # Column 4: index in output (spin box)
        index_spin = QSpinBox(self.input_table)
        index_spin.setRange(0, 1024)
        index_spin.setEnabled(False)
        index_spin.valueChanged.connect(
            lambda _val, r=row: self._on_input_mapping_changed(r)
        )
        self.input_table.setCellWidget(row, self.IN_MAP_INDEX_COL, index_spin)

    # ------------------------------------------------------------------
    # Output fields table setup
    # ------------------------------------------------------------------

    def _setup_default_output_fields(self) -> None:
        """
        Create the default output fields:
        - A 'coordinates' row (special, not a FieldType).
        - A '+' row to add new fields.
        """
        self.output_table.setRowCount(0)

        # Coordinates row (row 0), special
        self._insert_coordinates_row(0)

        # Add-row at bottom
        self._insert_add_row()

    def _insert_coordinates_row(self, row: int) -> None:
        """Insert the special coordinates row."""
        self.output_table.insertRow(row)

        # Name
        name_item = QTableWidgetItem("coordinates")
        # You *could* allow renaming; if you don't want that, clear the editable flag:
        # name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.output_table.setItem(row, self.OUT_NAME_COL, name_item)

        # Type column: fixed label "Coordinates"
        type_item = QTableWidgetItem("Coordinates")
        type_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.output_table.setItem(row, self.OUT_TYPE_COL, type_item)

        # #columns: fixed 3
        ncols_spin = QSpinBox(self.output_table)
        ncols_spin.setRange(3, 3)
        ncols_spin.setValue(3)
        ncols_spin.setEnabled(False)
        self.output_table.setCellWidget(row, self.OUT_NCOLS_COL, ncols_spin)

        # Button: none / disabled
        btn = QPushButton("", self.output_table)
        btn.setEnabled(False)
        self.output_table.setCellWidget(row, self.OUT_BTN_COL, btn)

    def _insert_output_field_row(
        self,
        row: int,
        name: str,
        ftype: FieldType,
        ncols: int,
    ) -> None:
        """
        Insert a 'real' (non-coordinates) output field row at the given index.
        """
        self.output_table.insertRow(row)

        # Column 0: name
        name_item = QTableWidgetItem(name)
        self.output_table.setItem(row, self.OUT_NAME_COL, name_item)

        # Column 1: field type (combo)
        type_combo = QComboBox(self.output_table)
        for ft in FieldType:
            type_combo.addItem(FieldType.get_human_name(ft), ft)
        idx = type_combo.findData(ftype)
        if idx >= 0:
            type_combo.setCurrentIndex(idx)
        type_combo.currentIndexChanged.connect(
            lambda _ix, r=row: self._on_output_type_changed(r)
        )
        self.output_table.setCellWidget(row, self.OUT_TYPE_COL, type_combo)

        # Column 2: # columns
        ncols_spin = QSpinBox(self.output_table)
        ncols_spin.setRange(1, 4096)
        ncols_spin.setValue(ncols)
        ncols_spin.valueChanged.connect(
            lambda val, r=row: self._on_output_ncols_changed(r, val)
        )
        self.output_table.setCellWidget(row, self.OUT_NCOLS_COL, ncols_spin)

        # Column 3: remove button
        btn = QPushButton("-", self.output_table)
        btn.clicked.connect(
            lambda _checked, r=row: self._remove_output_field_row(r)
        )
        self.output_table.setCellWidget(row, self.OUT_BTN_COL, btn)

        # Apply ncols rules based on type
        self._apply_type_rules_to_row(row)

    def _insert_add_row(self) -> None:
        """Insert the bottom '+' row."""
        row = self.output_table.rowCount()
        self.output_table.insertRow(row)

        for col in range(self.OUT_BTN_COL):
            empty_item = QTableWidgetItem("")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.output_table.setItem(row, col, empty_item)

        btn = QPushButton("+", self.output_table)
        btn.clicked.connect(self._on_add_field_clicked)
        self.output_table.setCellWidget(row, self.OUT_BTN_COL, btn)

    # ------------------------------------------------------------------
    # Output table helpers
    # ------------------------------------------------------------------

    def _real_output_row_count(self) -> int:
        """Number of 'real' rows (including coordinates, excluding '+' row)."""
        count = self.output_table.rowCount()
        return max(0, count - 1)

    def _is_add_row(self, row: int) -> bool:
        return row == self.output_table.rowCount() - 1

    def _is_coordinates_row(self, row: int) -> bool:
        return row == 0  # by construction

    def _on_add_field_clicked(self) -> None:
        """Handler for '+' button: add a new SCALAR field above it."""
        insert_row = self._real_output_row_count()  # insert above '+'
        name = self._generate_unique_scalar_name()
        self._insert_output_field_row(
            row=insert_row,
            name=name,
            ftype=FieldType.SCALAR,
            ncols=1,
        )
        self._refresh_input_mapping_targets()

    def _generate_unique_scalar_name(self) -> str:
        base = "ScalarField_"
        existing = {
            (cast(QTableWidgetItem,self.output_table.item(r, self.OUT_NAME_COL)).text()
             if self.output_table.item(r, self.OUT_NAME_COL)
             else "")
            for r in range(self._real_output_row_count())
            if not self._is_coordinates_row(r)
        }
        idx = 0
        while True:
            name = f"{base}{idx:03d}"
            if name not in existing:
                return name
            idx += 1

    def _remove_output_field_row(self, row: int) -> None:
        """Remove an output field row (not coordinates, not '+')."""
        if row < 0 or self._is_add_row(row) or self._is_coordinates_row(row):
            return

        name_item = self.output_table.item(row, self.OUT_NAME_COL)
        field_name = name_item.text() if name_item else None

        self.output_table.removeRow(row)

        # Ensure there's still a '+' row at the bottom
        if not self._is_add_row(self.output_table.rowCount() - 1):
            self._insert_add_row()

        # Clear any input mappings pointing to this field
        if field_name:
            self._clear_mappings_to_field(field_name)

        self._refresh_input_mapping_targets()

    def _clear_mappings_to_field(self, field_name: str) -> None:
        for row in range(self.input_table.rowCount()):
            combo: QComboBox = cast(QComboBox,self.input_table.cellWidget(
                row, self.IN_MAP_FIELD_COL
            ))
            if combo.currentText() == field_name:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                self._update_input_status_icon(row)

    def _apply_type_rules_to_row(self, row: int) -> None:
        """
        Enforce ncols rules based on the field type for non-coordinates rows.
        """
        if self._is_add_row(row) or self._is_coordinates_row(row):
            return

        type_combo: QComboBox = cast(QComboBox, self.output_table.cellWidget(
            row, self.OUT_TYPE_COL)
        )
        ncols_spin: QSpinBox = cast(QSpinBox, self.output_table.cellWidget(
            row, self.OUT_NCOLS_COL)
        )

        ftype: FieldType = type_combo.currentData()

        if ftype in (
            FieldType.SCALAR,
            FieldType.INTENSITY,
            FieldType.SEMANTIC,
            FieldType.INSTANCE,
        ):
            ncols_spin.setValue(1)
            ncols_spin.setEnabled(False)
        elif ftype == FieldType.COLOR:
            ncols_spin.setValue(3)
            ncols_spin.setEnabled(False)
        elif ftype == FieldType.VECTOR:
            ncols_spin.setEnabled(True)
            if ncols_spin.value() < 1:
                ncols_spin.setValue(1)

        self._update_index_enablement_for_all_inputs()

    # ------------------------------------------------------------------
    # Output table signal handlers
    # ------------------------------------------------------------------

    def _on_output_type_changed(self, row: int) -> None:
        if self._is_add_row(row) or self._is_coordinates_row(row):
            return
        self._apply_type_rules_to_row(row)
        self._refresh_input_mapping_targets()

    def _on_output_ncols_changed(self, row: int, val: int) -> None:
        if self._is_add_row(row) or self._is_coordinates_row(row):
            return

        type_combo: QComboBox = cast(QComboBox, self.output_table.cellWidget(
            row, self.OUT_TYPE_COL)
        )
        ftype: FieldType = type_combo.currentData()
        if ftype == FieldType.VECTOR and val > 128:
            QMessageBox.warning(
                self,
                "Large vector size",
                "You chose more than 128 columns for a VECTOR field.\n"
                "This may lead to a large memory footprint. "
                "Are you sure you want this?",
            )
        self._update_index_enablement_for_all_inputs()

    def _on_output_item_changed(self, item: QTableWidgetItem) -> None:
        """React to name changes in output fields (col 0)."""
        if self._suppress_output_item_changed:
            return
        if self._is_add_row(item.row()):
            return
        if item.column() == self.OUT_NAME_COL:
            self._refresh_input_mapping_targets()

    # ------------------------------------------------------------------
    # Input mapping helpers
    # ------------------------------------------------------------------

    def _refresh_input_mapping_targets(self) -> None:
        """Update the mapping combos for input fields."""
        # Get output field names (including coordinates, excluding '+')
        field_names: List[str] = []
        for r in range(self._real_output_row_count()):
            item = self.output_table.item(r, self.OUT_NAME_COL)
            if item:
                text = item.text().strip()
                if text:
                    field_names.append(text)

        for row in range(self.input_table.rowCount()):
            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                row, self.IN_MAP_FIELD_COL)
            )
            current = combo.currentText()

            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            for name in field_names:
                combo.addItem(name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

            self._on_input_mapping_changed(row)

    def _get_output_row_by_name(self, name: str) -> Optional[int]:
        for r in range(self._real_output_row_count()):
            item = self.output_table.item(r, self.OUT_NAME_COL)
            if item and item.text() == name:
                return r
        return None

    def _on_input_mapping_changed(self, row: int) -> None:
        self._update_index_enablement_for_row(row)
        self._update_input_status_icon(row)

    def _update_index_enablement_for_all_inputs(self) -> None:
        for r in range(self.input_table.rowCount()):
            self._update_index_enablement_for_row(r)
            self._update_input_status_icon(r)


    def _update_index_enablement_for_row(self, row: int) -> None: 
        """
        Enable/disable index spin based on mapped output field type:
        - Enabled for coordinates / VECTOR / COLOR
        - Disabled otherwise
        """
        combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
            row, self.IN_MAP_FIELD_COL )
        )
        index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
            row, self.IN_MAP_INDEX_COL)
        )

        field_name = combo.currentText().strip()
        if not field_name:
            index_spin.setEnabled(False)
            return

        out_row = self._get_output_row_by_name(field_name)
        if out_row is None:
            index_spin.setEnabled(False)
            return

        if self._is_coordinates_row(out_row):
            # coordinates: 3 columns, we always use indices 0..2
            ncols = 3
            index_spin.setEnabled(True)
            index_spin.setMaximum(ncols - 1)
            if index_spin.value() >= ncols:
                index_spin.setValue(ncols - 1)
            return

        # Non-coordinates row: check FieldType
        type_combo: QComboBox = cast(QComboBox, self.output_table.cellWidget(
            out_row, self.OUT_TYPE_COL)
        )
        ncols_spin: QSpinBox = cast(QSpinBox, self.output_table.cellWidget(
            out_row, self.OUT_NCOLS_COL)
        )

        ftype: FieldType = type_combo.currentData()
        ncols = ncols_spin.value()

        if ftype in (FieldType.VECTOR, FieldType.COLOR):
            index_spin.setEnabled(True)
            index_spin.setMaximum(max(0, ncols - 1))
            if index_spin.value() >= ncols:
                index_spin.setValue(ncols - 1)
        else:
            index_spin.setEnabled(False)

    def _update_input_status_icon(self, row: int) -> None:
        """
        Set '?' if unmapped/invalid, '✓' if valid mapping.
        """
        status_item = self.input_table.item(row, self.IN_STATUS_COL)
        combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
            row, self.IN_MAP_FIELD_COL)
        )
        index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
            row, self.IN_MAP_INDEX_COL)
        )

        field_name = combo.currentText().strip()
        if not field_name:
            cast(QTableWidgetItem, status_item).setText("?")
            return

        out_row = self._get_output_row_by_name(field_name)
        if out_row is None:
            cast(QTableWidgetItem, status_item).setText("?")
            return

        if self._is_coordinates_row(out_row):
            # coordinates: index must be valid
            if index_spin.isEnabled() and 0 <= index_spin.value() <= 2:
                cast(QTableWidgetItem, status_item).setText("✓")
            else:
                cast(QTableWidgetItem, status_item).setText("?")
            return

        type_combo: QComboBox = cast(QComboBox, self.output_table.cellWidget(
            out_row, self.OUT_TYPE_COL)
        )
        ncols_spin: QSpinBox = cast(QSpinBox, self.output_table.cellWidget(
            out_row, self.OUT_NCOLS_COL)
        )
        ftype: FieldType = type_combo.currentData()
        ncols = ncols_spin.value()

        if ftype in (FieldType.VECTOR, FieldType.COLOR):
            if not index_spin.isEnabled():
                cast(QTableWidgetItem, status_item).setText("?")
                return
            if 0 <= index_spin.value() < ncols:
                cast(QTableWidgetItem, status_item).setText("✓")
            else:
                cast(QTableWidgetItem, status_item).setText("?")
        else:
            # SCALAR / INTENSITY / SEMANTIC / INSTANCE: index ignored
            cast(QTableWidgetItem, status_item).setText("✓")

    # ------------------------------------------------------------------
    # Validation and acceptance
    # ------------------------------------------------------------------

    def accept(self) -> None:
        """
        Validate coordinates mapping and build PointCloudData.
        """
        if self.vertex_element is None:
            QMessageBox.critical(
                self, "Error", "No vertex data loaded from PLY."
            )
            return

        if not self._validate_coordinates_mapping():
            return

        try:
            self._point_cloud = self.build_point_cloud_data()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to construct PointCloudData:\n{e}",
            )
            return

        super().accept()

    def _validate_coordinates_mapping(self) -> bool:
        """
        Ensure that the coordinates row exists and indices 0,1,2 are mapped.
        """
        # Coordinates row is row 0 by construction
        if self.output_table.rowCount() < 2:
            QMessageBox.critical(
                self,
                "Missing coordinates",
                "No coordinates row defined.",
            )
            return False

        coord_item = self.output_table.item(0, self.OUT_NAME_COL)
        coord_name = coord_item.text() if coord_item else "coordinates"

        mapped_indices: List[int] = []
        for r in range(self.input_table.rowCount()):
            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                r, self.IN_MAP_FIELD_COL)
            )
            index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
                r, self.IN_MAP_INDEX_COL)
            )
            if combo.currentText().strip() == coord_name and index_spin.isEnabled():
                mapped_indices.append(index_spin.value())

        if set(mapped_indices) != {0, 1, 2}:
            QMessageBox.critical(
                self,
                "Incomplete coordinates mapping",
                "Coordinates must be fully mapped: indices 0, 1, and 2\n"
                "must each be assigned to input fields.",
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Public: get mappings (if you want raw mapping info)
    # ------------------------------------------------------------------

    def get_mappings(self) -> List[Dict[str, Any]]:
        """
        Returns a list of mappings for each input field.

        Each entry:
            - 'ply_field': str
            - 'dtype': str
            - 'output_field': Optional[str]   (name in output table)
            - 'output_index': Optional[int]   (index in that field, if used)
        """
        mappings: List[Dict[str, Any]] = []

        for r in range(self.input_table.rowCount()):
            name_item = self.input_table.item(r, self.IN_NAME_COL)
            dtype_item = self.input_table.item(r, self.IN_DTYPE_COL)
            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                r, self.IN_MAP_FIELD_COL)
            )
            index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
                r, self.IN_MAP_INDEX_COL)
            )

            ply_field = name_item.text() if name_item else ""
            dtype_str = dtype_item.text() if dtype_item else ""

            out_name = combo.currentText().strip()
            out_index = index_spin.value() if index_spin.isEnabled() else None

            mappings.append(
                {
                    "ply_field": ply_field,
                    "dtype": dtype_str,
                    "output_field": out_name or None,
                    "output_index": out_index,
                }
            )

        return mappings

    # ------------------------------------------------------------------
    # PointCloudData construction
    # ------------------------------------------------------------------

    def build_point_cloud_data(self) -> PointCloudData:
        """
        Construct a PointCloudData object based on the current mapping.
        """
        vertex_data = self.vertex_element.data # type:ignore
        num_points = len(vertex_data)

        # ---- Build coordinates (points) ---------------------------------
        coord_item = self.output_table.item(0, self.OUT_NAME_COL)
        coord_name = coord_item.text() if coord_item else "coordinates"

        # Map: index 0/1/2 -> ply field name
        coord_map: Dict[int, str] = {}
        for r in range(self.input_table.rowCount()):
            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                r, self.IN_MAP_FIELD_COL)
            )
            index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
                r, self.IN_MAP_INDEX_COL)
            )

            if combo.currentText().strip() == coord_name and index_spin.isEnabled():
                idx = index_spin.value()
                ply_field_item = self.input_table.item(r, self.IN_NAME_COL)
                if ply_field_item:
                    coord_map[idx] = ply_field_item.text()

        points = np.zeros((num_points, 3), dtype=np.float32)
        for dim in (0, 1, 2):
            field_name = coord_map.get(dim)
            if field_name is None:
                raise RuntimeError(f"Missing mapping for coordinate index {dim}.")
            values = np.asarray(vertex_data[field_name], dtype=np.float32)
            if values.shape[0] != num_points:
                raise ValueError(
                    f"Coordinate field '{field_name}' length mismatch: "
                    f"{values.shape[0]} vs {num_points}."
                )
            points[:, dim] = values

        pcd = PointCloudData(points)

        # ---- Build other fields -----------------------------------------
        mappings = self.get_mappings()

        # Collect fields: name -> (row_index, FieldType, ncols)
        output_fields: Dict[str, Any] = {}
        for r in range(1, self._real_output_row_count()):  # skip coordinates row
            name_item = self.output_table.item(r, self.OUT_NAME_COL)
            if not name_item:
                continue
            name = name_item.text().strip()
            if not name:
                continue

            type_widget = self.output_table.cellWidget(r, self.OUT_TYPE_COL)
            if not isinstance(type_widget, QComboBox):
                continue
            ftype: FieldType = type_widget.currentData()

            ncols_spin: QSpinBox = cast(QSpinBox, self.output_table.cellWidget(
                r, self.OUT_NCOLS_COL)
            )
            ncols = ncols_spin.value()

            output_fields[name] = (r, ftype, ncols)

        # For each field, allocate a data array and fill from mappings
        for out_name, (out_row, ftype, ncols) in output_fields.items():
            # Choose dtype
            if ftype in (FieldType.SEMANTIC, FieldType.INSTANCE):
                dtype = np.int32
            else:
                dtype = np.float32

            data = np.zeros((num_points, ncols), dtype=dtype)

            # Fill data from input mappings
            for i, m in enumerate(mappings):
                if m["output_field"] != out_name:
                    continue

                ply_field = m["ply_field"]
                src = np.asarray(vertex_data[ply_field])
                if src.shape[0] != num_points:
                    raise ValueError(
                        f"Field '{ply_field}' length mismatch: "
                        f"{src.shape[0]} vs {num_points}."
                    )

                # Flatten if needed
                if src.ndim > 1:
                    src = src.reshape(num_points)

                if ftype in (FieldType.VECTOR, FieldType.COLOR):
                    idx = m["output_index"]
                    if idx is None:
                        raise RuntimeError(
                            f"Missing index mapping for VECTOR/COLOR field '{out_name}'."
                        )
                    if idx < 0 or idx >= ncols:
                        raise RuntimeError(
                            f"Index {idx} out of range for field '{out_name}' with {ncols} columns."
                        )
                    data[:, idx] = src.astype(dtype, copy=False)
                else:
                    # SCALAR / INTENSITY / SEMANTIC / INSTANCE: single column
                    data[:, 0] = src.astype(dtype, copy=False)

            # Add field to PointCloudData
            if ftype == FieldType.SEMANTIC:
                # Pick a schema (here: first available, or default)
                schema = None
                if self.semantic_schemas:
                    schema = next(iter(self.semantic_schemas.values()))
                pcd.add_field(
                    name=out_name,
                    data=data,
                    field_type=FieldType.SEMANTIC,
                    schema=schema,
                )
            else:
                pcd.add_field(
                    name=out_name,
                    data=data,
                    field_type=ftype,
                    vector_dim_hint=ncols,
                )

                # Attach ColorMap for COLOR fields (via FieldBase.color_map)
                if ftype == FieldType.COLOR and self.color_maps:
                    cmap = next(iter(self.color_maps.values()))
                    field = pcd.get_fields(names=out_name)[0]
                    field.color_map = cmap

        return pcd
