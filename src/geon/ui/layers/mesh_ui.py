from copy import deepcopy
import numpy as np
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QDialog, QMenu, QMessageBox)
from geon.data.mesh import MeshLevel
from geon.rendering.mesh import MeshLayer
from geon.tools.command_manager import LambdaCommand
from .registry import LAYER_UI, LayerUIHooks


def edit_fields(layer, parent, controller):
    if layer.is_preview:
        QMessageBox.information(parent, "Mesh preview", "Accept the reconstruction before editing its fields.")
        return
    from geon.ui.imports import FieldEditorDialog
    level = layer.active_level
    container = layer.data.field_container(level)
    scratch = deepcopy(container)
    schemas = {field.schema.name: field.schema for field in layer.data.get_fields()
               if hasattr(field, "schema")}
    dataset_manager = getattr(parent.window(), "dataset_manager", None)
    dataset = getattr(dataset_manager, "_dataset", None)
    color_maps = {}
    if dataset is not None:
        schemas.update({schema.name: schema for schema in dataset.unique_semantic_schemas})
        color_maps = {mapping.name: mapping for mapping in dataset.unique_color_maps}
    dialog = FieldEditorDialog(ply_path=None, parent=parent, target_point_cloud=scratch, edit_only=True,
                               semantic_schemas=schemas, color_maps=color_maps)
    dialog.setWindowTitle(f"Edit mesh {level.value} fields")
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return
    before = deepcopy(container._fields)
    after = deepcopy(scratch._fields)
    def apply(fields):
        container._fields = deepcopy(fields)
        active = layer.active_field_name
        if not layer.data.get_fields(names=active, level=level):
            active = container.field_names[0] if container.field_names else None
        layer.set_active_field(level, active)
        controller.layer_data_modified.emit(layer)
        controller.scene_tree_request_change.emit()
        controller.ctx.viewer.rerender()
    controller.do_global(LambdaCommand("Edit mesh "+level.value+" fields", lambda: apply(after), lambda: apply(before)))


def _ribbon(layer, parent, controller):
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(2, 1, 2, 1)
    level_combo = QComboBox()
    field_combo = QComboBox()
    for level in MeshLevel:
        level_combo.addItem(f"{level.value.capitalize()} ({layer.data.element_count(level):,})", level)
    level_combo.setCurrentIndex(list(MeshLevel).index(layer.active_level))
    def fields():
        field_combo.blockSignals(True)
        field_combo.clear()
        field_combo.addItem("<geometry>", None)
        for field in layer.data.get_fields(level=layer.active_level):
            field_combo.addItem(field.name, field.name)
        field_combo.setCurrentIndex(max(0, field_combo.findData(layer.active_field_name)))
        field_combo.blockSignals(False)
    def changed_level():
        panel = getattr(parent.window(), "color_channels", None)
        if panel is not None and not panel.allow_switch():
            level_combo.setCurrentIndex(list(MeshLevel).index(layer.active_level))
            return
        level = level_combo.currentData()
        available = layer.data.get_fields(level=level)
        layer.set_active_field(level, available[0].name if available else None)
        fields()
        controller.layer_internal_sel_changed.emit(layer)
        controller.ctx.viewer.rerender()
    def changed_field():
        panel = getattr(parent.window(), "color_channels", None)
        if panel is not None and not panel.allow_switch():
            fields()
            return
        layer.set_active_field(layer.active_level, field_combo.currentData())
        controller.ctx.viewer.rerender()
    fields()
    level_combo.currentIndexChanged.connect(changed_level)
    field_combo.currentIndexChanged.connect(changed_field)
    layout.addWidget(level_combo)
    layout.addWidget(field_combo)
    edit = QPushButton("Edit fields")
    edit.clicked.connect(lambda: edit_fields(layer, parent, controller))
    layout.addWidget(edit)
    annotate = QPushButton("Classify")
    annotate.clicked.connect(lambda: controller.activate_tool("MeshAnnotateTool"))
    layout.addWidget(annotate)
    select = QPushButton("Select all")
    def select_all():
        old = layer.capture_selection()
        new = (layer.active_level, np.arange(layer.data.element_count(layer.active_level)))
        def apply(state):
            field = layer.active_field_name
            layer.restore_selection(state)
            if layer.data.get_fields(names=field, level=layer.active_level):
                layer.set_active_field(layer.active_level, field)
            controller.layer_internal_sel_changed.emit(layer)
            controller.ctx.viewer.rerender()
        controller.do_global(LambdaCommand("Select mesh elements", lambda: apply(new), lambda: apply(old)))
    select.clicked.connect(select_all)
    layout.addWidget(select)
    for control in (level_combo, field_combo, edit, annotate, select):
        control.setEnabled(not layer.is_preview)
    return widget


def _menu(layer, parent, controller):
    menu = QMenu(parent)
    menu.addAction("Edit fields", lambda: edit_fields(layer, parent, controller)).setEnabled(not layer.is_preview)
    menu.addAction("Classify", lambda: controller.activate_tool("MeshAnnotateTool")).setEnabled(not layer.is_preview)
    return menu


LAYER_UI.register(MeshLayer, LayerUIHooks(
    ribbon_widget=_ribbon,
    tree_menu=_menu,
    tree_item_text=lambda layer: layer.browser_name+(" (preview)" if layer.is_preview else ""),
))
