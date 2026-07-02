from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from geon.config.theme import UIStyle
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.tools.boundingbox import CreateHorizontalBoundingBoxTool
from geon.tools.controller import ToolController
from geon.util.resources import resource_path

from .registry import LAYER_UI, LayerUIHooks


def _ribbon(
    layer: BoundingBoxLayer,
    parent: QWidget,
    controller: ToolController,
) -> QWidget:
    w = QWidget(parent)
    outer = QHBoxLayout(w)
    outer.setAlignment(Qt.AlignmentFlag.AlignTop)
    outer.setContentsMargins(2, 1, 2, 1)
    outer.setSpacing(8)

    stats = QWidget(w)
    grid = QGridLayout(stats)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(4)
    grid.setVerticalSpacing(2)
    count = QLabel(f"Boxes: {layer.box_count:,}", stats)
    count.setStyleSheet(UIStyle.TYPE_LABEL.value)
    grid.addWidget(count, 0, 0)
    schema_name = layer.data.schema.name if layer.data.schema is not None else "None"
    schema = QLabel(f"Schema: {schema_name}", stats)
    schema.setStyleSheet(UIStyle.TYPE_LABEL.value)
    grid.addWidget(schema, 1, 0)
    stats.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
    outer.addWidget(stats)

    create_btn = QPushButton("Create horizontal box", w)
    create_btn.setIcon(QIcon(CreateHorizontalBoundingBoxTool.icon_path))
    create_btn.pressed.connect(lambda: controller.activate_tool(CreateHorizontalBoundingBoxTool.__name__))
    outer.addWidget(create_btn)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
    return w


def _ribbon_selection(
    layer: BoundingBoxLayer,
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
    return w


def _text(layer: BoundingBoxLayer) -> str:
    return layer.browser_name


def _icon(layer: BoundingBoxLayer) -> QIcon:
    return QIcon(resource_path("annotate.png"))


LAYER_UI.register(
    BoundingBoxLayer,
    LayerUIHooks(
        ribbon_widget=_ribbon,
        ribbon_sel_widget=_ribbon_selection,
        tree_item_text=_text,
        tree_item_icon=_icon,
    ),
)
