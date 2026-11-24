"""Structured controls for the ported primitive reconstruction method."""
from dataclasses import asdict, fields
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QWidget,
    QTabWidget, QScrollArea, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QPushButton, QLabel, QDialogButtonBox, QMessageBox)
from geon.data.pointcloud import FieldType
from geon.algorithms.reconstruction.core import Config
from geon.algorithms.reconstruction.transfer import default_config


GROUPS = {
    "Input and fitting": ["min_points", "subsample_distance", "exclude_ids", "alpha_radius",
        "max_plane_rmse", "fit_cylinders", "cylinder_trim_mode", "cylinder_plane_trigger",
        "cylinder_error_ratio", "max_cylinder_rmse", "cylinder_max_radius",
        "cylinder_min_angle_degrees", "curve_tolerance", "cylinder_chord_tolerance", "cylinder_mesh_spacing"],
    "B-spline fallback": ["fit_bsplines", "max_bspline_rmse", "bspline_error_ratio",
        "bspline_smoothing", "bspline_max_slope", "bspline_mesh_spacing"],
    "Borders and corners": ["clean_first", "adjacency_radius", "min_contact_points", "edge_tolerance", "support_step",
        "corner_tolerance", "min_edge_length", "min_plane_angle_degrees", "simplify_tolerance", "weld_tolerance"],
    "Coplanar joins": ["join_coplanar", "prefer_coplanar_join", "coplanar_join_mode", "coplanar_angle_degrees",
        "coplanar_plane_tolerance", "coplanar_boundary_tolerance", "regularize_coplanar_borders"],
    "Gap recovery": ["rescue_shared_edges", "bridge_remaining_gaps", "clean_border_pass", "stitch_mesh_boundaries",
        "mesh_stitch_tolerance", "junction_cap_tolerance", "gap_bridge_tolerance"],
}


class ReconstructionDialog(QDialog):
    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mesh from PCD instance field (experimental)")
        self.resize(650, 640)
        self.layer = layer
        self.config = None
        self.inputs = {}
        self.mapping_combos = []
        self.defaults = asdict(default_config())
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        source = QWidget()
        source_layout = QVBoxLayout(source)
        form = QFormLayout()
        source_layout.addLayout(form)
        self.name_input = QLineEdit(f"{layer.data.name} (reconstructed)")
        form.addRow("Mesh name", self.name_input)
        self.primitive_combo = QComboBox()
        instances = layer.data.get_fields(field_type=FieldType.INSTANCE)
        for field in instances:
            self.primitive_combo.addItem(field.name, field.name)
        if layer.active_field_name in [f.name for f in instances]:
            self.primitive_combo.setCurrentText(layer.active_field_name)
        form.addRow("Primitive segmentation", self.primitive_combo)
        self.selection_only = QCheckBox("Reconstruct selected points only")
        self.selection_only.setEnabled(layer.has_selection)
        form.addRow(self.selection_only)
        source_layout.addWidget(QLabel("Additional fields → face fields (majority for labels, mean for numeric values)."))
        self.mappings = QVBoxLayout()
        source_layout.addLayout(self.mappings)
        available = [f.name for f in layer.data.get_fields() if f.name != self.primitive_combo.currentData()]
        # Prefer semantic and instance fields for IFC classification and objects.
        available.sort(key=lambda name: layer.data.get_fields(names=name)[0].field_type not in
                       (FieldType.SEMANTIC, FieldType.INSTANCE))
        for index in range(2):
            self.add_mapping(available[index] if index < len(available) else None)
        add = QPushButton("Add mapped field")
        add.clicked.connect(lambda: self.add_mapping())
        source_layout.addWidget(add)
        preview_form = QFormLayout()
        self.preview_spacing = QDoubleSpinBox()
        self.preview_spacing.setDecimals(4)
        self.preview_spacing.setRange(.0001, 1000.)
        self.preview_spacing.setValue(.05)
        self.preview_spacing.setSuffix(" m")
        preview_form.addRow("Deviation sample spacing", self.preview_spacing)
        source_layout.addLayout(preview_form)
        source_layout.addStretch()
        tabs.addTab(source, "Fields")
        for title, names in GROUPS.items():
            widget = QWidget()
            form = QFormLayout(widget)
            for name in names:
                value = self.defaults[name]
                if isinstance(value, bool):
                    control = QCheckBox()
                    control.setChecked(value)
                elif name in ("coplanar_join_mode", "cylinder_trim_mode"):
                    control = QComboBox()
                    control.addItems(["bridge", "snap"] if name == "coplanar_join_mode" else ["observed", "extent"])
                    control.setCurrentText(value)
                elif name == "exclude_ids":
                    control = QLineEdit(", ".join(map(str, value)))
                elif isinstance(value, int):
                    control = QSpinBox()
                    control.setRange(1 if name == "min_contact_points" else 3, 100_000_000)
                    control.setValue(value)
                else:
                    control = QDoubleSpinBox()
                    control.setDecimals(7)
                    control.setRange(0., 1_000_000.)
                    control.setSingleStep(.01)
                    if name in ("subsample_distance", "corner_tolerance"):
                        control.setSpecialValueText("Automatic / disabled")
                    control.setValue(0. if value is None else value)
                self.inputs[name] = control
                if name == 'clean_first':
                    control.setToolTip('Clean individual outlines before matching complete seams and coordinating junctions. Disable to compare the previous pipeline.')
                form.addRow('Clean first (experimental)' if name == 'clean_first' else name.replace("_", " ").capitalize(), control)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            tabs.addTab(scroll, title)
        self.inputs['clean_first'].toggled.connect(self._update_border_controls)
        self._update_border_controls()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Reconstruct")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(instances))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _update_border_controls(self):
        legacy = not self.inputs['clean_first'].isChecked()
        for name in ('prefer_coplanar_join', 'rescue_shared_edges',
                     'bridge_remaining_gaps', 'clean_border_pass', 'gap_bridge_tolerance'):
            self.inputs[name].setEnabled(legacy)
            self.inputs[name].setToolTip('Used by the previous pipeline when Clean first is disabled.')

    def add_mapping(self, selected=None):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.addItem("<none>", None)
        for field in self.layer.data.get_fields():
            combo.addItem(f"{field.name} ({field.field_type.human_name})", field.name)
        if selected is not None:
            combo.setCurrentIndex(combo.findData(selected))
        remove = QPushButton("Remove")
        def drop():
            self.mapping_combos.remove(combo)
            self.mappings.removeWidget(row)
            row.deleteLater()
        remove.clicked.connect(drop)
        layout.addWidget(combo, 1)
        layout.addWidget(remove)
        self.mapping_combos.append(combo)
        self.mappings.addWidget(row)

    def mapped_names(self):
        return list(dict.fromkeys(combo.currentData() for combo in self.mapping_combos if combo.currentData()))

    def accept(self):
        try:
            values = dict(self.defaults)
            for name, control in self.inputs.items():
                if isinstance(control, QCheckBox):
                    value = control.isChecked()
                elif isinstance(control, QComboBox):
                    value = control.currentText()
                elif isinstance(control, QLineEdit):
                    value = tuple(int(part.strip()) for part in control.text().split(",") if part.strip())
                else:
                    value = control.value()
                    if name in ("subsample_distance", "corner_tolerance") and value == 0:
                        value = None
                values[name] = value
            if not self.name_input.text().strip():
                raise ValueError("Enter a mesh name.")
            if not self.primitive_combo.currentData():
                raise ValueError("Choose an instance field for primitive segmentation.")
            self.config = Config(**values)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Reconstruction settings", str(exc))
            return
        super().accept()
