"""IFC class mapping, JSON interchange and export controls."""
from copy import deepcopy
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QTableWidget, QTableWidgetItem, QPushButton, QDialogButtonBox,
    QFileDialog, QMessageBox, QHeaderView, QProgressDialog)
import numpy as np
from geon.data.mesh import MeshLevel
from geon.data.pointcloud import FieldType
from geon.io.ifc import (IFC_SCHEMA, available_ifc_classes, predefined_types, default_mapping,
                          load_mapping, save_mapping, prepare_objects, export_ifc)


class IfcExportDialog(QDialog):
    def __init__(self, mesh, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export mesh to IFC 4.3 (experimental)")
        self.resize(850, 650)
        self.mesh = mesh
        self.mapping = None
        self.defaults = deepcopy(mesh.metadata.get("ifc_map") or default_mapping())
        self.classes = available_ifc_classes()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.semantic_combo = QComboBox()
        self.object_combo = QComboBox()
        self.face_combo = QComboBox()
        self.object_combo.addItem("<mesh objects — object-level semantics only>", None)
        self.face_combo.addItem("<none>", None)
        for level in MeshLevel:
            for field in mesh.get_fields(level=level):
                reference = {"level": level.value, "name": field.name}
                label = f"{level.value}/{field.name}"
                if field.field_type == FieldType.SEMANTIC:
                    self.semantic_combo.addItem(label, reference)
                if field.field_type == FieldType.INSTANCE:
                    self.object_combo.addItem(label, reference)
                    self.face_combo.addItem(label, reference)
        form.addRow("Semantic field", self.semantic_combo)
        form.addRow("Object separation", self.object_combo)
        form.addRow("Face groups (optional)", self.face_combo)
        layout.addLayout(form)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Semantic class", "IFC4.3 class", "Predefined type"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(3, 175)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        load = QPushButton("Load ifc_map.json")
        load.clicked.connect(self._load)
        save = QPushButton("Save mapping…")
        save.clicked.connect(self._save)
        row.addWidget(load)
        row.addWidget(save)
        row.addStretch()
        layout.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.semantic_combo.currentIndexChanged.connect(self._semantic_changed)
        self._set_reference(self.semantic_combo, self.defaults.get("semantic_field"))
        self._set_reference(self.object_combo, self.defaults.get("object_field"))
        if self.object_combo.currentIndex() <= 0 and self.object_combo.count() > 1:
            self.object_combo.setCurrentIndex(1)
        self._set_reference(self.face_combo, self.defaults.get("face_field"))
        self._semantic_changed()

    @staticmethod
    def _set_reference(combo, reference):
        for index in range(combo.count()):
            if combo.itemData(index) == reference:
                combo.setCurrentIndex(index)
                return True
        return False

    def _semantic_changed(self):
        from geon.io.ifc import resolve_field
        reference = self.semantic_combo.currentData()
        self.table.setRowCount(0)
        if not reference:
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        field = resolve_field(self.mesh, reference, FieldType.SEMANTIC)
        self.object_combo.setToolTip("Required when semantic labels are not at object level.")
        existing = self.defaults.get("classes", {})
        by_name = {item.get("name"): item for item in existing.values()}
        names = {int(cls.id): cls.name for cls in field.schema.semantic_classes}
        for value in np.unique(field.data):
            names.setdefault(int(value), f"Unlabeled ({value})")
        self._rows = []
        for row, (class_id, name) in enumerate(sorted(names.items())):
            self.table.insertRow(row)
            for column, value in ((0, str(class_id)), (1, name)):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            definition = by_name.get(name, {"ifc_class": "IfcBuildingElementProxy"})
            class_combo = QComboBox()
            class_combo.setEditable(True)
            class_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            class_combo.addItems(self.classes)
            class_combo.setCurrentText(definition.get("ifc_class", "IfcBuildingElementProxy"))
            type_combo = QComboBox()
            def refresh_types(index=None, combo=class_combo, types=type_combo):
                current = types.currentText()
                types.clear()
                types.addItem("<default>", None)
                if combo.currentText() in self.classes:
                    types.addItems(predefined_types(combo.currentText()))
                target = types.findText(current)
                types.setCurrentIndex(max(0, target))
            class_combo.currentTextChanged.connect(refresh_types)
            refresh_types()
            index = type_combo.findText(definition.get("predefined_type", ""))
            if index >= 0:
                type_combo.setCurrentIndex(index)
            self.table.setCellWidget(row, 2, class_combo)
            self.table.setCellWidget(row, 3, type_combo)
            self._rows.append((class_id, name, class_combo, type_combo))

    def current_mapping(self):
        from geon.io.ifc import resolve_field
        reference = self.semantic_combo.currentData()
        field = resolve_field(self.mesh, reference, FieldType.SEMANTIC)
        classes = {}
        for class_id, name, combo, types in self._rows:
            if combo.currentText() not in self.classes:
                raise ValueError(f"Choose an IFC4.3 class for {name}.")
            definition = {"name": name, "ifc_class": combo.currentText()}
            if types.currentData() is not None or types.currentIndex() > 0:
                definition["predefined_type"] = types.currentText()
            classes[str(class_id)] = definition
        return {"version": 1, "schema": IFC_SCHEMA, "semantic_schema": field.schema.name,
                "semantic_field": reference, "object_field": self.object_combo.currentData(),
                "face_field": self.face_combo.currentData(), "classes": classes}

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load IFC mapping", "", "JSON files (*.json)")
        if not path:
            return
        try:
            mapping = load_mapping(path)
            if not self._set_reference(self.semantic_combo, mapping.get("semantic_field")):
                raise ValueError("The mapped semantic field is not present in this mesh.")
            for combo, key in ((self.object_combo, "object_field"), (self.face_combo, "face_field")):
                if not self._set_reference(combo, mapping.get(key)):
                    raise ValueError(f"The mapped {key.replace('_', ' ')} is not present in this mesh.")
            self.defaults = mapping
            self._semantic_changed()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Load IFC mapping", str(exc))

    def _save(self):
        try:
            mapping = self.current_mapping()
            path, _ = QFileDialog.getSaveFileName(self, "Save IFC mapping", "ifc_map.json", "JSON files (*.json)")
            if path:
                save_mapping(mapping, path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save IFC mapping", str(exc))

    def accept(self):
        try:
            self.mapping = self.current_mapping()
            prepare_objects(self.mesh, self.mapping)
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "IFC mapping", str(exc))
            return
        super().accept()


def export_mesh_with_dialog(mesh, parent):
    try:
        dialog = IfcExportDialog(mesh, parent)
    except (ImportError, RuntimeError, ValueError) as exc:
        QMessageBox.warning(parent, "IFC export", str(exc))
        return None
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    path, _ = QFileDialog.getSaveFileName(parent, "Export mesh to IFC", mesh.name+".ifc", "IFC files (*.ifc)")
    if not path:
        return None
    errors = []
    class Worker(QThread):
        def run(self):
            try:
                export_ifc(mesh, path, dialog.mapping)
            except Exception as exc:
                errors.append(str(exc))
    progress = QProgressDialog("Exporting IFC 4.3…", "", 0, 0, parent)
    progress.setWindowTitle("IFC export")
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    worker = Worker(progress)
    worker.finished.connect(progress.accept)
    worker.start()
    progress.exec()
    worker.wait()
    if errors:
        QMessageBox.warning(parent, "IFC export failed", errors[0])
        return None
    return dialog.mapping
