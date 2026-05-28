from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QPushButton, QSizePolicy, QWidget

from geon.config.theme import UIStyle
from geon.rendering.cellcomplex import CellComplexLayer
from geon.tools.cellcomplex import CellComplexAddEdgeTool, CellComplexAssociateTool
from geon.tools.controller import ToolController
from geon.util.resources import resource_path

from .registry import LAYER_UI, LayerUIHooks


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
    outer.addWidget(add_edge_btn)
    associate_btn = QPushButton("Associate", w)
    associate_btn.setIcon(QIcon(CellComplexAssociateTool.icon_path))
    associate_btn.pressed.connect(lambda: controller.activate_tool(CellComplexAssociateTool.__name__))
    outer.addWidget(associate_btn)
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
