from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QLabel,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from geon.config.theme import UIStyle
from geon.data.pointcloud import SemanticSchema
from geon.rendering.cellcomplex import CellComplexLayer
from geon.tools.cellcomplex import (
    CellComplexAddEdgeTool,
    CellComplexAnnotateTool,
    CellComplexAssociateTool,
    cell_complex_annotation_target,
)
from geon.tools.controller import ToolController
from geon.ui.cellcomplex_dialogs import CreateCellSemanticAttributeDialog
from geon.util.resources import resource_path

from .registry import LAYER_UI, LayerUIHooks


def _collect_semantic_schemas(layer: CellComplexLayer, parent: QWidget) -> list[SemanticSchema]:
    window = parent.window()
    dataset_manager = getattr(window, "dataset_manager", None) if window is not None else None
    dataset = getattr(dataset_manager, "_dataset", None) if dataset_manager is not None else None
    if dataset is not None:
        return list(dataset.unique_semantic_schemas)
    schemas: list[SemanticSchema] = []
    seen: set[tuple] = set()
    for _dim, _name, schema in layer.data.iter_semantic_attributes():
        signature = (schema.name, schema.signature())
        if signature in seen:
            continue
        seen.add(signature)
        schemas.append(schema)
    return schemas


def _create_semantic_attribute(
    layer: CellComplexLayer,
    parent: QWidget,
    controller: ToolController,
) -> None:
    schemas = _collect_semantic_schemas(layer, parent)
    taken_schema_names = sorted({schema.name for schema in schemas})
    dlg = CreateCellSemanticAttributeDialog(
        schemas=schemas,
        taken_schema_names=taken_schema_names,
        parent=parent,
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    if dlg.attribute_name is None or dlg.dimension is None or dlg.schema is None:
        return
    try:
        layer.data.add_semantic_attribute(dlg.dimension, dlg.attribute_name, dlg.schema)
        layer.update()
    except ValueError as exc:
        QMessageBox.warning(parent, "Cannot create semantic attribute", str(exc))
        return
    available_names = layer.semantic_attribute_names(dlg.dimension)
    if dlg.attribute_name not in available_names:
        cell_count = len(layer.data.get_cells(dlg.dimension))
        QMessageBox.warning(
            parent,
            "Cannot activate semantic attribute",
            f"Created semantic attribute '{dlg.attribute_name}', but it is not available "
            f"for dimension {dlg.dimension}. Cells in dimension: {cell_count}. "
            f"Available attributes: {', '.join(available_names) or 'none'}.",
        )
        return
    try:
        layer.set_active_semantic_attribute(dlg.dimension, dlg.attribute_name)
    except ValueError as exc:
        QMessageBox.warning(parent, "Cannot activate semantic attribute", str(exc))
        return
    if controller.ctx is not None:
        controller.ctx.controller.layer_internal_sel_changed.emit(layer)
        controller.ctx.controller.scene_tree_request_change.emit()
        mark_modified = getattr(controller.ctx.viewer.window(), "_mark_active_doc_modified", None)
        if callable(mark_modified):
            mark_modified()
        controller.ctx.viewer.rerender()


def _button_columns(buttons: list[QPushButton], parent: QWidget) -> QWidget:
    w = QWidget(parent)
    grid = QGridLayout(w)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(4)
    grid.setVerticalSpacing(2)
    widest = max((btn.sizeHint().width() for btn in buttons), default=0)
    for index, btn in enumerate(buttons):
        btn.setMinimumWidth(widest)
        grid.addWidget(btn, index % 2, index // 2)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
    return w


def _ribbon(
    layer: CellComplexLayer,
    parent: QWidget,
    controller: ToolController,
) -> QWidget:
    w = QWidget(parent)
    outer = QHBoxLayout(w)
    outer.setAlignment(Qt.AlignmentFlag.AlignTop)
    outer.setContentsMargins(2, 1, 2, 1)
    outer.setSpacing(8)

    stats = QLabel(
        f"Vertices: {layer.vertex_count:,}   Edges: {layer.edge_count:,}   "
        f"Size: {layer.size_mode}"
    )
    stats.setStyleSheet(UIStyle.TYPE_LABEL.value)
    outer.addWidget(stats)

    add_edge_btn = QPushButton("Add edge", w)
    add_edge_btn.setIcon(QIcon(CellComplexAddEdgeTool.icon_path))
    add_edge_btn.pressed.connect(lambda: controller.activate_tool(CellComplexAddEdgeTool.__name__))
    create_attr_btn = QPushButton("Create semantic attribute", w)
    create_attr_btn.setIcon(QIcon(resource_path("tree_icon_field.png")))
    create_attr_btn.pressed.connect(lambda: _create_semantic_attribute(layer, parent, controller))
    associate_btn = QPushButton("Associate", w)
    associate_btn.setIcon(QIcon(CellComplexAssociateTool.icon_path))
    associate_btn.pressed.connect(lambda: controller.activate_tool(CellComplexAssociateTool.__name__))
    outer.addWidget(_button_columns([add_edge_btn, create_attr_btn, associate_btn], w))
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
    return w


def _ribbon_selection(
    layer: CellComplexLayer,
    parent: QWidget,
    controller: ToolController,
) -> QWidget | None:
    if not layer.active_selection:
        return None
    w = QWidget(parent)
    outer = QHBoxLayout(w)
    outer.setContentsMargins(2, 1, 2, 1)
    label = QLabel(f"Size: {layer.browser_sel_descr}", w)
    label.setStyleSheet(UIStyle.TYPE_LABEL.value)
    outer.addWidget(label)
    deselect_btn = QPushButton("Deselect", w)
    deselect_btn.setIcon(QIcon(resource_path("deselect.png")))
    deselect_btn.pressed.connect(lambda: controller.activate_tool("DeselectTool"))
    outer.addWidget(deselect_btn)
    annotate_btn = QPushButton("Annotate", w)
    annotate_btn.setIcon(QIcon(CellComplexAnnotateTool.icon_path))
    annotate_btn.setEnabled(cell_complex_annotation_target(layer) is not None)
    annotate_btn.pressed.connect(lambda: controller.activate_tool(CellComplexAnnotateTool.__name__))
    outer.addWidget(annotate_btn)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
    return w


def _text(layer: CellComplexLayer) -> str:
    return layer.browser_name


def _icon(layer: CellComplexLayer) -> QIcon:
    return QIcon(resource_path("tree_icon_cellcomplex.png"))


LAYER_UI.register(
    CellComplexLayer,
    LayerUIHooks(
        ribbon_widget=_ribbon,
        ribbon_sel_widget=_ribbon_selection,
        tree_item_text=_text,
        tree_item_icon=_icon,
    ),
)
