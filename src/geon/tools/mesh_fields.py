"""Classification of mesh vertices, edges, faces or objects."""
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar
import weakref
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox
from geon.data.pointcloud import FieldType
from geon.rendering.mesh import MeshLayer
from .base import ModeTool
from .command_manager import LambdaCommand


@dataclass
class MeshAnnotateTool(ModeTool):
    label: ClassVar = "mesh_annotate"
    tooltip: ClassVar = "Classify mesh elements"
    shortcut: ClassVar = None
    icon_path: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar = False
    show_in_toolbar: ClassVar = False
    keep_focus: ClassVar = False

    def activate(self):
        super().activate()

    def create_context_widget(self, parent):
        widget = QWidget(parent)
        layout = QHBoxLayout(widget)
        layer = self.ctx.scene.active_layer
        if not isinstance(layer, MeshLayer):
            layout.addWidget(QLabel("Activate a mesh first."))
            return widget
        self.layer = layer
        self.level = layer.active_level
        layout.addWidget(QLabel(self.level.value.capitalize()+":"))
        self.fields = QComboBox()
        for field in layer.data.get_fields(level=self.level):
            if field.field_type in (FieldType.SEMANTIC, FieldType.INSTANCE):
                self.fields.addItem(field.name, field)
        if layer.active_field_name:
            index = self.fields.findText(layer.active_field_name)
            if index >= 0:
                self.fields.setCurrentIndex(index)
        self.values = QComboBox()
        self.all_elements = QCheckBox("All elements")
        self.accept_button = QPushButton("Accept (S)")
        self.accept_button.clicked.connect(self._accept)
        self.fields.currentIndexChanged.connect(self._refresh)
        layout.addWidget(self.fields)
        layout.addWidget(self.values)
        layout.addWidget(self.all_elements)
        layout.addWidget(self.accept_button)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.ctx.controller.deactivate_tool)
        layout.addWidget(cancel)
        self._refresh()
        return widget

    def _refresh(self):
        self.values.clear()
        field = self.fields.currentData()
        if field is not None:
            if field.field_type == FieldType.SEMANTIC:
                for cls in field.schema.semantic_classes:
                    self.values.addItem(cls.name, cls.id)
            else:
                self.values.addItem("New instance", "new")
                self.values.addItem("Unassigned", -1)
        self.accept_button.setEnabled(self.values.count() > 0)

    def _accept(self):
        layer = self.layer
        field = self.fields.currentData()
        if field is None or self.values.currentData() is None:
            return
        if self.all_elements.isChecked():
            selected = np.arange(layer.data.element_count(self.level))
        elif layer.active_level == self.level:
            selected = layer.active_selection
        else:
            return
        if selected is None or len(selected) == 0:
            return
        selected = selected.copy()
        previous = field.data[selected].copy()
        value = self.values.currentData()
        if value == "new":
            value = field.get_next_instance_id()
        layer_ref = weakref.ref(layer)
        field_name = field.name
        level = self.level
        def apply(values):
            target = layer_ref()
            if target is None:
                return
            fields = target.data.get_fields(names=field_name, level=level)
            if not fields:
                return
            fields[0].data[selected] = values
            target.update()
            self.ctx.controller.layer_data_modified.emit(target)
            self.ctx.viewer.rerender()
        self.command_manager.do(LambdaCommand("Classify mesh "+self.level.value,
                                             lambda: apply(value), lambda: apply(previous)))
        self.ctx.controller.deactivate_tool()

    def key_press_hook(self, event):
        if event.key and event.key.lower() == "escape":
            self.ctx.controller.deactivate_tool()
        elif event.key and event.key.lower() == "s" and not (event.ctrl or event.alt):
            self._accept()
