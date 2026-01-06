from __future__ import annotations

from typing import Dict, Any, Optional, List, cast
import os.path as osp

import numpy as np
from plyfile import PlyData

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,
    QSplitter,QWidget,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,
    QComboBox,QPushButton,QSpinBox,QDialogButtonBox,QMessageBox,QCheckBox,QSizePolicy,
    QToolButton, QFileDialog
)

from PyQt6.QtGui import QColor

from geon.data.pointcloud import (
    PointCloudData,
    FieldType,
    SemanticSchema,
    SemanticSegmentation,
    InstanceSegmentation,
    FieldBase,
)
from geon.data.definitions import ColorMap  
from geon.ui.semantic_schema_dialog import SemanticSchemaCreationDialog


class FieldEditorDialog(QDialog):
    """
    Dialog that maps input fields into PointCloudData fields.
    Supports PLY import, generic input arrays, and edit-only modes.
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
    OUT_RESOURCE_COL = 3
    OUT_BTN_COL = 4 # script assumes the OUT_BTN_COL is always last
    
    NEW_SCHEMA_SENTINEL = "__NEW_SCHEMA__"
    SOURCE_NAME_ROLE = Qt.ItemDataRole.UserRole
    SOURCE_KIND_ROLE = Qt.ItemDataRole.UserRole + 1
    AUTO_INPUT_FIELD_MAPPINGS: Dict[str, tuple[Optional[str], FieldType, int]] = {
        # colors 
        "red": ("colors", FieldType.COLOR, 0),
        "r": ("colors", FieldType.COLOR, 0),
        "green": ("colors", FieldType.COLOR, 1),
        "g": ("colors", FieldType.COLOR, 1),
        "blue": ("colors", FieldType.COLOR, 2),
        "b": ("colors", FieldType.COLOR, 2),
        # normals 
        "nx": ("normals", FieldType.NORMAL, 0),
        "ny": ("normals", FieldType.NORMAL, 1),
        "nz": ("normals", FieldType.NORMAL, 2),
        "normalx": ("normals", FieldType.NORMAL, 0),
        "normaly": ("normals", FieldType.NORMAL, 1),
        "normalz": ("normals", FieldType.NORMAL, 2),
        "normal_x": ("normals", FieldType.NORMAL, 0),
        "normal_y": ("normals", FieldType.NORMAL, 1),
        "normal_z": ("normals", FieldType.NORMAL, 2),
        # intensity-like scalars 
        "intensity": ("intensity", FieldType.INTENSITY, 0),
        "intensities": ("intensity", FieldType.INTENSITY, 0),
        "reflectance": ("intensity", FieldType.INTENSITY, 0),
        "reflectivity": ("intensity", FieldType.INTENSITY, 0),
        "luminosity": ("intensity", FieldType.INTENSITY, 0),
        "luminance": ("intensity", FieldType.INTENSITY, 0),
        # semantic segmentation 
        "label": (None, FieldType.SEMANTIC, 0),
        "labels": (None, FieldType.SEMANTIC, 0),
        "class": (None, FieldType.SEMANTIC, 0),
        "classes": (None, FieldType.SEMANTIC, 0),
        "semantic": (None, FieldType.SEMANTIC, 0),
        "semantic_label": (None, FieldType.SEMANTIC, 0),
        "ground_truth": (None, FieldType.SEMANTIC, 0),
        "groundtruth": (None, FieldType.SEMANTIC, 0),
        "gt": (None, FieldType.SEMANTIC, 0),
        "seg": (None, FieldType.SEMANTIC, 0),
        "segmentation": (None, FieldType.SEMANTIC, 0),
        "semantic_segmentation": (None, FieldType.SEMANTIC, 0),
        # instance segmentation 
        "instance": (None, FieldType.INSTANCE, 0),
        "instance_id": (None, FieldType.INSTANCE, 0),
        "instanceid": (None, FieldType.INSTANCE, 0),
        "object_id": (None, FieldType.INSTANCE, 0),
        "objectid": (None, FieldType.INSTANCE, 0),
        "inst": (None, FieldType.INSTANCE, 0),
    }
    AUTO_INPUT_COORD_MAPPINGS: Dict[str, int] = {
        "x": 0,
        "y": 1,
        "z": 2,
        "posx": 0,
        "posy": 1,
        "posz": 2,
        "positionx": 0,
        "positiony": 1,
        "positionz": 2,
        "coordx": 0,
        "coordy": 1,
        "coordz": 2,
    }

    def __init__(
        self,
        ply_path: Optional[str],
        semantic_schemas: Dict[str, SemanticSchema],
        color_maps: Dict[str, ColorMap],
        allow_doc_appending = False,
        taken_doc_names: list[str] = [],
        parent: Optional[QWidget] = None,
        input_fields: Optional[Dict[str, np.ndarray]] = None,
        input_field_dtypes: Optional[Dict[str, str]] = None,
        input_source_name: Optional[str] = None,
        target_point_cloud: Optional[PointCloudData] = None,
        edit_only: bool = False,
    ) -> None:
        super().__init__(parent)

        self.ply_path = ply_path
        self.semantic_schemas = semantic_schemas
        self.color_maps = color_maps
        self._target_pointcloud = target_point_cloud
        self._edit_only = edit_only
        self._allow_coordinate_mapping = self._target_pointcloud is None and not self._edit_only
        self._allow_field_duplicate = self._edit_only or self._target_pointcloud is not None
        self._input_fields: Dict[str, np.ndarray] = {}
        self._input_field_dtypes: Dict[str, str] = {}
        self._input_field_names: list[str] = []
        self._input_source_name = input_source_name or ("PLY" if ply_path else "Input")

        if self._edit_only:
            self.setWindowTitle("Edit Fields")
        else:
            self.setWindowTitle(f"Import {self._input_source_name}")
        self.resize(1100, 650)

        self._suppress_output_item_changed = False
        self._point_cloud: Optional[PointCloudData] = None
        self._create_new_doc_flag : bool = True
        self._allow_doc_appending : bool = allow_doc_appending
        self._taken_doc_names: list[str] = taken_doc_names
        # self._schema_by_output_row: dict[int, Optional[SemanticSchema]] = {}

        if input_fields is not None:
            self._set_input_fields(input_fields, input_field_dtypes)
        elif self.ply_path:
            self._load_ply()
        self._build_ui()
        self._populate_input_fields()
        self._setup_default_output_fields()
        self._apply_auto_mappings()

    @staticmethod
    def load_npy_fields(npy_path: str) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
        arr = np.load(npy_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            raise ValueError("Unsupported format: expected .npy, got .npz archive.")
        arr = np.asarray(arr)

        base = osp.splitext(osp.basename(npy_path))[0].strip()
        if not base:
            base = "field"

        def _column_names_for_base(base_name: str, ncols: int) -> list[str]:
            name = base_name.strip().lower()
            if ncols == 3:
                if "normal" in name:
                    return ["nx", "ny", "nz"]
                if name in {"color", "colors", "rgb"} or "color" in name:
                    return ["red", "green", "blue"]
            return [f"{base_name}_{i}" for i in range(ncols)]

        if arr.ndim == 1:
            fields = {base: arr}
        elif arr.ndim == 2:
            if arr.shape[1] == 1:
                fields = {base: arr.reshape(-1)}
            else:
                names = _column_names_for_base(base, arr.shape[1])
                fields = {names[i]: arr[:, i] for i in range(arr.shape[1])}
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}. Expected 1D or 2D array.")

        dtypes = {name: str(np.asarray(val).dtype) for name, val in fields.items()}
        return fields, dtypes

    @classmethod
    def from_npy_picker(
        cls,
        parent: QWidget,
        semantic_schemas: Dict[str, SemanticSchema],
        color_maps: Dict[str, ColorMap],
        target_point_cloud: Optional[PointCloudData],
    ) -> Optional["FieldEditorDialog"]:
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Open NumPy File", "", "NumPy Files (*.npy)"
        )
        if not file_path:
            return None
        try:
            input_fields, input_dtypes = cls.load_npy_fields(file_path)
        except Exception as exc:
            QMessageBox.critical(
                parent,
                "Import .npy failed",
                f"Could not load NumPy file:\n{exc}",
            )
            return None
        return cls(
            ply_path=None,
            semantic_schemas=semantic_schemas,
            color_maps=color_maps,
            input_fields=input_fields,
            input_field_dtypes=input_dtypes,
            input_source_name="NPY",
            target_point_cloud=target_point_cloud,
            parent=parent,
        )

    # ------------------------------------------------------------------
    # Accessor for the resulting PointCloudData
    # ------------------------------------------------------------------

    @property
    def point_cloud(self) -> Optional[PointCloudData]:
        return self._point_cloud
    @property
    def create_new_doc_flag(self) -> bool:
        return self._create_new_doc_flag

    # ------------------------------------------------------------------
    # PLY loading
    # ------------------------------------------------------------------

    def _load_ply(self) -> None:
        """Load the PLY file and store the vertex element."""
        if not self.ply_path:
            self.vertex_element = None
            return

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
        vertex_data = self.vertex_element.data
        names = vertex_data.dtype.names or []
        input_fields = {name: np.asarray(vertex_data[name]) for name in names}
        input_dtypes = {
            name: str(vertex_data.dtype.fields[name][0]) for name in names
            if vertex_data.dtype.fields.get(name)
        }
        self._set_input_fields(input_fields, input_dtypes, field_order=list(names))

    def _set_input_fields(
        self,
        fields: Dict[str, np.ndarray],
        dtypes: Optional[Dict[str, str]] = None,
        field_order: Optional[list[str]] = None,
    ) -> None:
        self._input_fields = {name: np.asarray(arr) for name, arr in fields.items()}
        if field_order is None:
            self._input_field_names = list(self._input_fields.keys())
        else:
            self._input_field_names = [n for n in field_order if n in self._input_fields]
        if dtypes is None:
            self._input_field_dtypes = {
                name: str(self._input_fields[name].dtype)
                for name in self._input_field_names
            }
        else:
            self._input_field_dtypes = {
                name: dtypes.get(name, str(self._input_fields[name].dtype))
                for name in self._input_field_names
            }

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _make_resource_combo(self) -> QComboBox:
        combo = QComboBox(self.output_table)
        combo.addItem("_") # placeholder
        combo.setEnabled(False)
        return combo
    
    def _populate_resource_combo_for_row(self, row:int) -> None:
        if self._is_add_row(row) or self._is_coordinates_row(row):
            return
        
        type_combo = self.output_table.cellWidget(row, self.OUT_TYPE_COL)
        res_combo = self.output_table.cellWidget(row, self.OUT_RESOURCE_COL)
        
        if not isinstance(type_combo, QComboBox) or not isinstance(res_combo, QComboBox):
            return
        
        ftype: FieldType = type_combo.currentData()
        
        prev_key = res_combo.currentData()
        res_combo.blockSignals(True)
        res_combo.clear()
        
        if ftype == FieldType.COLOR:
            res_combo.setEnabled(False)
            res_combo.addItem("-", None)
        
        elif ftype == FieldType.SEMANTIC:
            res_combo.setEnabled(True)
            res_combo.addItem("Empty", None)
            for key, schema in self.semantic_schemas.items():
                res_combo.addItem(schema.name, key)
            res_combo.addItem("<New Schema>", self.NEW_SCHEMA_SENTINEL)
            ix = res_combo.count() - 1
            res_combo.setItemData(ix, QColor("dodgerblue"), Qt.ItemDataRole.ForegroundRole)
            res_combo.currentIndexChanged.connect(lambda _ix, r=row: self._on_resource_changed(r))
            
            
                
        else:
            res_combo.setEnabled(False)
            res_combo.addItem("-", None)
            
        # restore selection if present
        if prev_key is not None:
            ix = res_combo.findData(prev_key)
            if ix >= 0:
                res_combo.setCurrentIndex(ix)
            else:
                res_combo.setCurrentIndex(0)
                
        res_combo.blockSignals(False)
            
    def _get_ply_fields_mapped_to_output(self, output_field_name: str) -> list[str]:
        mapped: list[str] = []
        for r in range(self.input_table.rowCount()):
            name_item = self.input_table.item(r, self.IN_NAME_COL)
            ply_name = name_item.text() if name_item else ""
            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(r, self.IN_MAP_FIELD_COL))
            if combo.currentText().strip() == output_field_name and ply_name:
                mapped.append(ply_name)
        return mapped       
    
    def _required_ids_from_mapped_semantic(self, output_row: int) -> list[int]:
        # Always include -1 as obligatory
        required: set[int] = {-1}

        name_item = self.output_table.item(output_row, self.OUT_NAME_COL)
        out_name = name_item.text().strip() if name_item else ""
        if not out_name or not self._input_fields:
            return sorted(required)

        ply_fields = self._get_ply_fields_mapped_to_output(out_name)

        if not ply_fields:
            return sorted(required)

        # For SEMANTIC, you typically expect exactly one mapped PLY field.
        # If multiple are mapped, we union them.
        for ply_field in ply_fields:
            if ply_field not in self._input_fields:
                continue
            arr = np.asarray(self._input_fields[ply_field])

            # Flatten: PLY fields can be (N,) or (N,1) etc.
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0])

            # Convert to int and collect unique values
            # (use np.unique for speed)
            try:
                vals = np.unique(arr.astype(np.int64, copy=False))
            except Exception:
                continue

            for v in vals.tolist():
                required.add(int(v))

        return sorted(required)

        
    def _on_resource_changed(self, row: int) -> None:
        if self._is_add_row(row) or self._is_coordinates_row(row):
            return

        type_combo = self.output_table.cellWidget(row, self.OUT_TYPE_COL)
        res_combo  = self.output_table.cellWidget(row, self.OUT_RESOURCE_COL)
        if not isinstance(type_combo, QComboBox) or not isinstance(res_combo, QComboBox):
            return

        ftype: FieldType = type_combo.currentData()
        data = res_combo.currentData()

        if ftype == FieldType.SEMANTIC and data == self.NEW_SCHEMA_SENTINEL:
            # Required IDs: for now default [-1]; later you can pass more.
            required_ids = self._required_ids_from_mapped_semantic(row)
            dlg = SemanticSchemaCreationDialog(
                required_ids=required_ids,
                taken_schema_names=list(self.semantic_schemas.keys()),
                parent=self,
            )
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.schema is not None:
                schema = dlg.schema  # SemanticSchema
                # Insert it into semantic_schemas (choose a key strategy)
                key = schema.name
                if key in self.semantic_schemas:
                    QMessageBox.warning(
                        self,
                        "Schema name exists",
                        "Schema name already exists. Choose a different name.",
                    )
                    self._populate_resource_combo_for_row(row)
                    res_combo.setCurrentIndex(0)
                    return

                self.semantic_schemas[key] = schema

                # repopulate this row combo and select the new schema
                self._populate_resource_combo_for_row(row)
                ix = res_combo.findData(key)
                if ix >= 0:
                    res_combo.setCurrentIndex(ix)
            else:
                # user cancelled: revert to Empty
                self._populate_resource_combo_for_row(row)
                res_combo.setCurrentIndex(0)

    
    # def _make_schema_cell_widget(self, row: int) -> QWidget:
    #     w = QWidget(self.output_table)
    #     lay = QHBoxLayout(w)
    #     lay.setContentsMargins(6,0,6,0)
    #     lay.setSpacing(6)
        
        
    #     label = QLabel(w)
    #     label.setObjectName('schemaLabel')
    #     label.setText("Empty")
    #     label.setStyleSheet('color: rgba(0,0,0,120);')
    #     label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
    #     btn = QPushButton("...", w)
    #     btn.setObjectName('schemaBtn')
    #     btn.setFixedWidth(28)
    #     # btn.clicked.connect(lambda _checked=False, r=row: self._on_choose_schema_clicked(r))
        
    #     lay.addWidget(label, 1)
    #     lay.addWidget(btn,0)
    #     return w
    
    
        
        
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        show_input = bool(self._input_fields) and not self._edit_only

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

        self.output_table = QTableWidget(0, 5, self)
        self.output_table.setHorizontalHeaderLabels(
            ["Name", "Field type", "# columns", "Schema", ""]
        )
        cast(QHeaderView, self.output_table.horizontalHeader()).setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.output_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.output_table.itemChanged.connect(self._on_output_item_changed)

        if show_input:
            splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(splitter)

            # Left: Input fields
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_label = QLabel(f"Input fields ({self._input_source_name})")
            left_label.setStyleSheet("font-weight: bold;")
            left_layout.addWidget(left_label)
            left_layout.addWidget(self.input_table)

            # Right: Output fields
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_label = QLabel("Point cloud data")
            right_label.setStyleSheet("font-weight: bold;")
            right_layout.addWidget(right_label)
            right_layout.addWidget(self.output_table)

            splitter.addWidget(left_widget)
            splitter.addWidget(right_widget)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
        else:
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_label = QLabel("Point cloud data")
            right_label.setStyleSheet("font-weight: bold;")
            right_layout.addWidget(right_label)
            right_layout.addWidget(self.output_table)
            main_layout.addWidget(right_widget)

        # Buttons
        bottom_layout = QHBoxLayout()

        self.create_new_doc_checkbox = QCheckBox("Create new document", self)
        self.create_new_doc_checkbox.setChecked(self._create_new_doc_flag)
        self.create_new_doc_checkbox.toggled.connect(self._on_create_new_doc_toggled)
        if not self._allow_doc_appending:
            self.create_new_doc_checkbox.setChecked(True)
            self.create_new_doc_checkbox.setEnabled(False)
        if self._target_pointcloud is not None or self._edit_only:
            self.create_new_doc_checkbox.setChecked(False)
            self.create_new_doc_checkbox.setEnabled(False)
            self.create_new_doc_checkbox.setVisible(False)
        
        bottom_layout.addWidget(self.create_new_doc_checkbox)

        bottom_layout.addStretch()  # push buttons to the right

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bottom_layout.addWidget(buttons)

        main_layout.addLayout(bottom_layout)

    def _on_create_new_doc_toggled(self, checked: bool) -> None:
        self._create_new_doc_flag = checked


    # ------------------------------------------------------------------
    # Populate input table from PLY
    # ------------------------------------------------------------------

    def _populate_input_fields(self) -> None:
        self.input_table.setRowCount(0)
        if not self._input_fields:
            return

        for name in self._input_field_names:
            dtype_str = self._input_field_dtypes.get(name, "")
            self._add_input_row(name, dtype_str)
            
    def _normalize_auto_field_name(self, name: str) -> str:
        return name.strip().lower()

    def _default_ncols_for_type(self, ftype: FieldType) -> int:
        if ftype in (FieldType.COLOR, FieldType.NORMAL):
            return 3
        if ftype in (
            FieldType.SCALAR,
            FieldType.INTENSITY,
            FieldType.SEMANTIC,
            FieldType.INSTANCE,
        ):
            return 1
        return 3

    def _apply_auto_mappings(self) -> None:
        """
        Create output fields and input mappings based on AUTO_INPUT_FIELD_MAPPINGS.
        """
        if self._edit_only or not self._input_fields:
            return

        output_needs: Dict[str, tuple[FieldType, set[int]]] = {}
        for row in range(self.input_table.rowCount()):
            name_item = self.input_table.item(row, self.IN_NAME_COL)
            if not name_item:
                continue
            input_name = name_item.text().strip()
            key = self._normalize_auto_field_name(input_name)
            mapping = self.AUTO_INPUT_FIELD_MAPPINGS.get(key)
            if mapping is None:
                continue
            out_name, ftype, out_index = mapping
            if out_name is None:
                out_name = input_name
            if out_name not in output_needs:
                output_needs[out_name] = (ftype, set())
            output_needs[out_name][1].add(out_index)

        for out_name, (ftype, _indices) in output_needs.items():
            out_row = self._get_output_row_by_name(out_name)
            if out_row is not None:
                continue
            insert_row = self._real_output_row_count()
            self._insert_output_field_row(
                row=insert_row,
                name=out_name,
                ftype=ftype,
                ncols=self._default_ncols_for_type(ftype),
            )

        self._refresh_input_mapping_targets()

        for row in range(self.input_table.rowCount()):
            name_item = self.input_table.item(row, self.IN_NAME_COL)
            if not name_item:
                continue
            input_name = name_item.text().strip()
            key = self._normalize_auto_field_name(input_name)
            coord_index = self.AUTO_INPUT_COORD_MAPPINGS.get(key)
            if coord_index is not None and self._allow_coordinate_mapping:
                combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                    row, self.IN_MAP_FIELD_COL
                ))
                index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
                    row, self.IN_MAP_INDEX_COL
                ))
                combo.blockSignals(True)
                coord_idx = combo.findText("coordinates")
                if coord_idx >= 0:
                    combo.setCurrentIndex(coord_idx)
                combo.blockSignals(False)
                self._update_index_enablement_for_row(row)
                index_spin.blockSignals(True)
                index_spin.setValue(coord_index)
                index_spin.blockSignals(False)
                self._on_input_mapping_changed(row)
                continue

            mapping = self.AUTO_INPUT_FIELD_MAPPINGS.get(key)
            if mapping is None:
                continue
            out_name, _ftype, out_index = mapping
            if out_name is None:
                out_name = input_name

            combo: QComboBox = cast(QComboBox, self.input_table.cellWidget(
                row, self.IN_MAP_FIELD_COL
            ))
            index_spin: QSpinBox = cast(QSpinBox, self.input_table.cellWidget(
                row, self.IN_MAP_INDEX_COL
            ))
            combo.blockSignals(True)
            idx = combo.findText(out_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            self._update_index_enablement_for_row(row)
            if index_spin.isEnabled():
                index_spin.blockSignals(True)
                index_spin.setValue(out_index)
                index_spin.blockSignals(False)
            self._on_input_mapping_changed(row)

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
        if self._target_pointcloud is not None:
            self._setup_output_fields_from_point_cloud(self._target_pointcloud)
            return

        self.output_table.setRowCount(0)

        # Coordinates row (row 0), special
        self._insert_coordinates_row(0, editable=self._allow_coordinate_mapping)

        # Add-row at bottom
        self._insert_add_row()

    def _setup_output_fields_from_point_cloud(self, pcd: PointCloudData) -> None:
        self.output_table.setRowCount(0)
        self._insert_coordinates_row(0, editable=False)

        insert_row = 1
        for field in pcd.get_fields():
            data = field.data
            ncols = 1 if data.ndim == 1 else data.shape[1]
            res_key = field.schema.name if isinstance(field, SemanticSegmentation) else None
            self._insert_output_field_row(
                row=insert_row,
                name=field.name,
                ftype=field.field_type,
                ncols=ncols,
                source_field_name=field.name,
                source_kind="existing",
                res_key=res_key,
            )
            insert_row += 1

        self._insert_add_row()

    def _insert_coordinates_row(self, row: int, editable: bool = True) -> None:
        """Insert the special coordinates row."""
        self.output_table.insertRow(row)

        # Name
        name_item = QTableWidgetItem("coordinates")
        if not editable:
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
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

        # resources
        res_combo = self._make_resource_combo()
        self.output_table.setCellWidget(row, self.OUT_RESOURCE_COL, res_combo)
                
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
        source_field_name: Optional[str] = None,
        source_kind: str = "new",
        res_key: Optional[str] = None,
    ) -> None:
        """
        Insert a 'real' (non-coordinates) output field row at the given index.
        """
        self.output_table.insertRow(row)

        # Column 0: name
        name_item = QTableWidgetItem(name)
        name_item.setData(self.SOURCE_NAME_ROLE, source_field_name)
        name_item.setData(self.SOURCE_KIND_ROLE, source_kind)
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

        # Column 3: schema / colormap selector
        res_combo = self._make_resource_combo()
        self.output_table.setCellWidget(row, self.OUT_RESOURCE_COL, res_combo)
        
        
        # Column 4: row actions
        self.output_table.setCellWidget(row, self.OUT_BTN_COL, self._make_row_action_widget())

        # Apply ncols rules based on type
        self._apply_type_rules_to_row(row)
        if res_key is not None:
            res_combo = self.output_table.cellWidget(row, self.OUT_RESOURCE_COL)
            if isinstance(res_combo, QComboBox):
                ix = res_combo.findData(res_key)
                if ix >= 0:
                    res_combo.setCurrentIndex(ix)

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

    def _row_from_widget(self, widget: QWidget) -> int:
        pos = widget.mapTo(self.output_table.viewport(), QPoint(1, 1))
        return self.output_table.indexAt(pos).row()

    def _make_row_action_widget(self) -> QWidget:
        w = QWidget(self.output_table)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if self._allow_field_duplicate:
            dup_btn = QToolButton(w)
            dup_btn.setText("Dup")
            dup_btn.clicked.connect(
                lambda _checked=False, btn=dup_btn: self._duplicate_output_field_row(
                    self._row_from_widget(btn)
                )
            )
            layout.addWidget(dup_btn)

        del_btn = QToolButton(w)
        del_btn.setText("Del")
        del_btn.clicked.connect(
            lambda _checked=False, btn=del_btn: self._remove_output_field_row(
                self._row_from_widget(btn)
            )
        )
        layout.addWidget(del_btn)

        return w

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
        base = "Field_"
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

    def _generate_unique_field_name(self, base: str) -> str:
        existing = {
            (cast(QTableWidgetItem, self.output_table.item(r, self.OUT_NAME_COL)).text()
             if self.output_table.item(r, self.OUT_NAME_COL)
             else "")
            for r in range(self._real_output_row_count())
            if not self._is_coordinates_row(r)
        }
        if base not in existing:
            return base
        idx = 1
        while True:
            name = f"{base}_{idx:03d}"
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

    def _duplicate_output_field_row(self, row: int) -> None:
        if row < 0 or self._is_add_row(row) or self._is_coordinates_row(row):
            return

        name_item = self.output_table.item(row, self.OUT_NAME_COL)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        type_combo = self.output_table.cellWidget(row, self.OUT_TYPE_COL)
        ncols_spin = self.output_table.cellWidget(row, self.OUT_NCOLS_COL)
        if not isinstance(type_combo, QComboBox) or not isinstance(ncols_spin, QSpinBox):
            return

        ftype: FieldType = type_combo.currentData()
        ncols = ncols_spin.value()
        source_name = name_item.data(self.SOURCE_NAME_ROLE) or name
        new_name = self._generate_unique_field_name(f"{name}_copy")
        insert_row = min(row + 1, self._real_output_row_count())
        self._insert_output_field_row(
            row=insert_row,
            name=new_name,
            ftype=ftype,
            ncols=ncols,
            source_field_name=source_name,
            source_kind="duplicate",
        )
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
        elif ftype in (FieldType.COLOR, FieldType.NORMAL):
            ncols_spin.setValue(3)
            ncols_spin.setEnabled(False)
        elif ftype == FieldType.VECTOR:
            ncols_spin.setEnabled(True)
            if ncols_spin.value() < 1:
                ncols_spin.setValue(1)

        self._update_index_enablement_for_all_inputs()
        self._populate_resource_combo_for_row(row)

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
        if self._edit_only or not self._input_fields:
            return
        # Get output field names (including coordinates, excluding '+')
        field_names: List[str] = []
        for r in range(self._real_output_row_count()):
            if self._is_coordinates_row(r) and not self._allow_coordinate_mapping:
                continue
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
        - Enabled for coordinates / VECTOR / COLOR / NORMAL
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

        if ftype in (FieldType.VECTOR, FieldType.COLOR, FieldType.NORMAL):
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

        if ftype in (FieldType.VECTOR, FieldType.COLOR, FieldType.NORMAL):
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

    def _validate_output_fields(self) -> bool:
        names: set[str] = set()
        for r in range(self._real_output_row_count()):
            if self._is_coordinates_row(r):
                continue
            item = self.output_table.item(r, self.OUT_NAME_COL)
            name = item.text().strip() if item else ""
            if not name:
                QMessageBox.critical(
                    self,
                    "Invalid field name",
                    "Field names cannot be empty.",
                )
                return False
            if name == "points":
                QMessageBox.critical(
                    self,
                    "Invalid field name",
                    "Field name 'points' is reserved.",
                )
                return False
            if name in names:
                QMessageBox.critical(
                    self,
                    "Duplicate field name",
                    f"Field name '{name}' is used more than once.",
                )
                return False
            names.add(name)
        return True

    def accept(self) -> None:
        """
        Validate coordinates mapping and build PointCloudData.
        """
        if self._edit_only:
            if self._target_pointcloud is None:
                QMessageBox.critical(
                    self, "Error", "No target point cloud provided."
                )
                return
            if not self._validate_output_fields():
                return
            try:
                self._apply_output_fields_to_point_cloud(self._target_pointcloud)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to update fields:\n{e}",
                )
                return
            self._point_cloud = self._target_pointcloud
            super().accept()
            return

        if self._target_pointcloud is not None:
            if not self._validate_output_fields():
                return
            try:
                self._apply_mappings_to_point_cloud(self._target_pointcloud)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to update fields:\n{e}",
                )
                return
            self._point_cloud = self._target_pointcloud
            super().accept()
            return

        if not self._input_fields:
            QMessageBox.critical(
                self, "Error", "No input data loaded."
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
        if not self._allow_coordinate_mapping:
            return True
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
    # Public: get mappings 
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

    def _field_ncols(self, field: FieldBase) -> int:
        data = field.data
        return 1 if data.ndim == 1 else data.shape[1]

    def _collect_output_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for r in range(self._real_output_row_count()):
            if self._is_coordinates_row(r):
                continue
            name_item = self.output_table.item(r, self.OUT_NAME_COL)
            type_widget = self.output_table.cellWidget(r, self.OUT_TYPE_COL)
            ncols_spin = self.output_table.cellWidget(r, self.OUT_NCOLS_COL)
            if not name_item or not isinstance(type_widget, QComboBox) or not isinstance(ncols_spin, QSpinBox):
                continue
            name = name_item.text().strip()
            ftype: FieldType = type_widget.currentData()
            ncols = ncols_spin.value()
            res_combo = self.output_table.cellWidget(r, self.OUT_RESOURCE_COL)
            res_key = res_combo.currentData() if isinstance(res_combo, QComboBox) else None
            source_name = name_item.data(self.SOURCE_NAME_ROLE)
            source_kind = name_item.data(self.SOURCE_KIND_ROLE) or "new"
            specs.append(
                {
                    "name": name,
                    "ftype": ftype,
                    "ncols": ncols,
                    "res_key": res_key,
                    "source_name": source_name,
                    "source_kind": source_kind,
                }
            )
        return specs

    def _collect_mappings_by_output(self) -> Dict[str, List[Dict[str, Any]]]:
        by_output: Dict[str, List[Dict[str, Any]]] = {}
        for m in self.get_mappings():
            out_name = m["output_field"]
            if out_name:
                by_output.setdefault(out_name, []).append(m)
        return by_output

    def _build_output_data_from_mappings(
        self,
        out_name: str,
        ftype: FieldType,
        ncols: int,
        num_points: int,
        mappings_by_output: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[np.ndarray]:
        mappings = mappings_by_output.get(out_name)
        if not mappings:
            return None

        dtype = np.int32 if ftype in (FieldType.SEMANTIC, FieldType.INSTANCE) else np.float32
        data = np.zeros((num_points, ncols), dtype=dtype)
        for m in mappings:
            ply_field = m["ply_field"]
            if ply_field not in self._input_fields:
                raise ValueError(f"Missing input field '{ply_field}'.")
            src = np.asarray(self._input_fields[ply_field])
            if src.shape[0] != num_points:
                raise ValueError(
                    f"Field '{ply_field}' length mismatch: {src.shape[0]} vs {num_points}."
                )
            if src.ndim > 1:
                src = src.reshape(src.shape[0])

            if ftype in (FieldType.VECTOR, FieldType.COLOR, FieldType.NORMAL):
                idx = m["output_index"]
                if idx is None:
                    raise RuntimeError(
                        f"Missing index mapping for VECTOR/COLOR/NORMAL field '{out_name}'."
                    )
                if idx < 0 or idx >= ncols:
                    raise RuntimeError(
                        f"Index {idx} out of range for field '{out_name}' with {ncols} columns."
                    )
                data[:, idx] = src.astype(dtype, copy=False)
            else:
                data[:, 0] = src.astype(dtype, copy=False)

        return data

    def _assign_field_data(self, field: FieldBase, data: np.ndarray, ftype: FieldType) -> None:
        if ftype in (FieldType.SEMANTIC, FieldType.INSTANCE):
            field.data = np.asarray(data, np.int32).reshape(-1)
        else:
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            field.data = arr

    def _make_field_for_spec(
        self,
        spec: Dict[str, Any],
        data: Optional[np.ndarray],
        num_points: int,
        source_field: Optional[FieldBase],
    ) -> FieldBase:
        ftype: FieldType = spec["ftype"]
        name = spec["name"]
        ncols = spec["ncols"]
        schema = None
        if ftype == FieldType.SEMANTIC:
            res_key = spec.get("res_key")
            if res_key == self.NEW_SCHEMA_SENTINEL:
                res_key = None
            if res_key is not None:
                schema = self.semantic_schemas.get(res_key, SemanticSchema())
            elif isinstance(source_field, SemanticSegmentation):
                schema = source_field.schema

        if data is None:
            if ftype == FieldType.SEMANTIC:
                return SemanticSegmentation(name, size=num_points, schema=schema)
            if ftype == FieldType.INSTANCE:
                return InstanceSegmentation(name, size=num_points)
            arr = np.zeros((num_points, ncols), dtype=np.float32)
            field = FieldBase(name, arr, ftype)
        else:
            if ftype == FieldType.SEMANTIC:
                return SemanticSegmentation(name, data=data, schema=schema)
            if ftype == FieldType.INSTANCE:
                return InstanceSegmentation(name, data=data)
            arr = np.asarray(data, np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            field = FieldBase(name, arr, ftype)

        if ftype == FieldType.COLOR and isinstance(source_field, FieldBase):
            field.color_map = source_field.color_map
        return field

    def _apply_output_specs_to_point_cloud(
        self,
        pcd: PointCloudData,
        mappings_by_output: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        specs = self._collect_output_specs()
        existing_by_name = {f.name: f for f in pcd.get_fields()}
        new_fields: List[FieldBase] = []
        num_points = pcd.points.shape[0]

        for spec in specs:
            source_kind = spec.get("source_kind", "new")
            source_name = spec.get("source_name") if source_kind in ("existing", "duplicate") else None
            source_field = existing_by_name.get(source_name) if source_name else None

            mapped_data = None
            if mappings_by_output is not None:
                mapped_data = self._build_output_data_from_mappings(
                    spec["name"],
                    spec["ftype"],
                    spec["ncols"],
                    num_points,
                    mappings_by_output,
                )

            if mapped_data is not None and source_field is not None:
                if (source_field.field_type == spec["ftype"] and
                        self._field_ncols(source_field) == spec["ncols"]):
                    self._assign_field_data(source_field, mapped_data, spec["ftype"])
                    source_field.name = spec["name"]
                    new_fields.append(source_field)
                    continue

            if mapped_data is None:
                if source_kind == "existing" and source_field is not None:
                    if (source_field.field_type == spec["ftype"] and
                            self._field_ncols(source_field) == spec["ncols"]):
                        source_field.name = spec["name"]
                        new_fields.append(source_field)
                        continue
                if source_kind == "duplicate" and source_field is not None:
                    dup_data = np.array(source_field.data, copy=True)
                    new_fields.append(
                        self._make_field_for_spec(spec, dup_data, num_points, source_field)
                    )
                    continue

            new_fields.append(
                self._make_field_for_spec(spec, mapped_data, num_points, source_field)
            )

        pcd._fields = new_fields

    def _apply_output_fields_to_point_cloud(self, pcd: PointCloudData) -> None:
        self._apply_output_specs_to_point_cloud(pcd, mappings_by_output=None)

    def _apply_mappings_to_point_cloud(self, pcd: PointCloudData) -> None:
        if not self._input_fields:
            raise RuntimeError("No input data loaded.")
        mappings_by_output = self._collect_mappings_by_output()
        num_points = pcd.points.shape[0]
        used_fields = {
            m["ply_field"]
            for mappings in mappings_by_output.values()
            for m in mappings
        }
        for name in used_fields:
            if name not in self._input_fields:
                raise ValueError(f"Missing input field '{name}'.")
            arr = self._input_fields[name]
            if arr.shape[0] != num_points:
                raise ValueError(
                    f"Input field '{name}' length mismatch: {arr.shape[0]} vs {num_points}."
                )
        self._apply_output_specs_to_point_cloud(pcd, mappings_by_output=mappings_by_output)

    # ------------------------------------------------------------------
    # PointCloudData construction
    # ------------------------------------------------------------------

    def build_point_cloud_data(self) -> PointCloudData:
        """
        Construct a PointCloudData object based on the current mapping.
        """
        if not self._input_fields:
            raise RuntimeError("No input data loaded.")
        num_points: Optional[int] = None

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

        points = None
        for dim in (0, 1, 2):
            field_name = coord_map.get(dim)
            if field_name is None:
                raise RuntimeError(f"Missing mapping for coordinate index {dim}.")
            if field_name not in self._input_fields:
                raise RuntimeError(f"Missing input field '{field_name}'.")
            values = np.asarray(self._input_fields[field_name], dtype=np.float32)
            if num_points is None:
                num_points = values.shape[0]
                points = np.zeros((num_points, 3), dtype=np.float32)
            if values.shape[0] != num_points:
                raise ValueError(
                    f"Coordinate field '{field_name}' length mismatch: "
                    f"{values.shape[0]} vs {num_points}."
                )
            assert points is not None
            points[:, dim] = values

        if points is None or num_points is None:
            raise RuntimeError("Failed to build coordinates.")
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

            res_combo = self.output_table.cellWidget(r, self.OUT_RESOURCE_COL)
            res_key = res_combo.currentData() if isinstance(res_combo, QComboBox) else None
            output_fields[name] = r, ftype, ncols, res_key

        # For each field, allocate a data array and fill from mappings
        for out_name, (out_row, ftype, ncols, res_key) in output_fields.items():
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
                if ply_field not in self._input_fields:
                    raise ValueError(f"Missing input field '{ply_field}'.")
                src = np.asarray(self._input_fields[ply_field])
                if src.shape[0] != num_points:
                    raise ValueError(
                        f"Field '{ply_field}' length mismatch: "
                        f"{src.shape[0]} vs {num_points}."
                    )

                # Flatten if needed
                if src.ndim > 1:
                    src = src.reshape(num_points)

                if ftype in (FieldType.VECTOR, FieldType.COLOR, FieldType.NORMAL):
                    idx = m["output_index"]
                    if idx is None:
                        raise RuntimeError(
                            f"Missing index mapping for VECTOR/COLOR/NORMAL field '{out_name}'."
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
                if res_key is not None:
                    schema = self.semantic_schemas.get(res_key, SemanticSchema())
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

                # Attach ColorMap for COLOR fields (via FieldBase.color_map) FIXME: fix color maps
                if ftype == FieldType.COLOR and self.color_maps:
                    cmap = next(iter(self.color_maps.values()))
                    field = pcd.get_fields(names=out_name)[0]
                    field.color_map = cmap

        return pcd
