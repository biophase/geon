from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import weakref

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from geon.data.cellcomplex import EdgeCell, PointCloudRef
from geon.data.pointcloud import FieldType, InstanceSegmentation
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.util.resources import resource_path

from .base import Event, ModeTool
from .command_manager import Command
from .tool_context import ToolContext


def _refresh_context(ctx: ToolContext, layer: CellComplexLayer) -> None:
    layer.update()
    ctx.controller.layer_internal_sel_changed.emit(layer)
    ctx.controller.scene_tree_request_change.emit()
    mark_modified = getattr(ctx.viewer.window(), "_mark_active_doc_modified", None)
    if callable(mark_modified):
        mark_modified()
    ctx.viewer.rerender()


@dataclass
class SelectCellComplexCellsCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    selection_new: set[str] | None
    selection_old: set[str] | None = None

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        if self.selection_old is None:
            self.selection_old = set(layer.active_selection or ())
        layer.active_selection = set(self.selection_new or ())
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.active_selection = set(self.selection_old or ())
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()


@dataclass
class DeleteCellComplexCellsCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    vertex_ids: set[str]
    edge_ids: set[str]
    _removed_vertices: list = field(default_factory=list, init=False)
    _removed_edges: list[EdgeCell] = field(default_factory=list, init=False)
    _selection_old: set[str] = field(default_factory=set, init=False)

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        self._selection_old = set(layer.active_selection or ())
        removed_edges: list[EdgeCell] = []
        if self.edge_ids:
            removed_edges.extend(layer.data.remove_edges(set(self.edge_ids)))
        if self.vertex_ids:
            vertices, incident_edges = layer.data.remove_vertices(set(self.vertex_ids))
            self._removed_vertices = vertices
            removed_edges.extend(edge for edge in incident_edges if edge.id not in {e.id for e in removed_edges})
        self._removed_edges = removed_edges
        layer.active_selection = set(layer.active_selection or ()) - self.vertex_ids - self.edge_ids - {e.id for e in removed_edges}
        _refresh_context(ctx, layer)

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        existing_vertex_ids = {vertex.id for vertex in layer.data.vertices}
        for vertex in self._removed_vertices:
            if vertex.id not in existing_vertex_ids:
                layer.data.vertices.append(vertex)
                existing_vertex_ids.add(vertex.id)
        existing_edge_ids = {edge.id for edge in layer.data.edges}
        for edge in self._removed_edges:
            if edge.id not in existing_edge_ids:
                layer.data.edges.append(edge)
                existing_edge_ids.add(edge.id)
        layer.active_selection = set(self._selection_old)
        _refresh_context(ctx, layer)


@dataclass
class CreateCellComplexEdgeCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    vertex_id_a: str
    vertex_id_b: str
    _edge: EdgeCell | None = None

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        if self._edge is None:
            self._edge = layer.data.build_edge(self.vertex_id_a, self.vertex_id_b)
        elif self._edge.id not in {edge.id for edge in layer.data.edges}:
            layer.data.append_edge(self._edge)
        _refresh_context(ctx, layer)

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None or self._edge is None:
            return
        layer.data.remove_edges({self._edge.id})
        layer.active_selection = set(layer.active_selection or ()) - {self._edge.id}
        _refresh_context(ctx, layer)


@dataclass
class AddCellGeometryRefCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    cell_id: str
    ref: PointCloudRef

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        cell = layer.data.get_cell_by_id(self.cell_id)
        if cell is None:
            return
        if self.ref not in cell.geometry_refs:
            cell.geometry_refs.append(self.ref)
        _refresh_context(ctx, layer)

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        cell = layer.data.get_cell_by_id(self.cell_id)
        if cell is None:
            return
        cell.geometry_refs = [ref for ref in cell.geometry_refs if ref is not self.ref]
        _refresh_context(ctx, layer)


@dataclass
class CellPickVertexTool(ModeTool):
    label: ClassVar = "cell_pick_vertex"
    tooltip: ClassVar = "Pick cell vertices"
    icon_path: ClassVar = resource_path("classify_selection.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    def left_button_release_hook(self, event: Event) -> None:
        layer = self.ctx.scene.active_layer
        if not isinstance(layer, CellComplexLayer):
            return
        layer.handle_viewport_left_click(self.ctx, event, self.ctx.viewer.pick(prefer_cells=True))

    def activate(self) -> None:
        return super().activate()


@dataclass
class CellComplexAddEdgeTool(ModeTool):
    label: ClassVar = "cell_complex_add_edge"
    tooltip: ClassVar = "Add CellComplex edge"
    icon_path: ClassVar = resource_path("add_ce_edge_tool.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    _label: QLabel | None = field(default=None, init=False)
    _create_btn: QPushButton | None = field(default=None, init=False)

    def _layer(self) -> CellComplexLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, CellComplexLayer) else None

    def _selected_vertex_ids(self) -> list[str]:
        layer = self._layer()
        return layer.selected_ids_by_dim(0) if layer is not None else []

    def _update_context_ui(self) -> None:
        ids = self._selected_vertex_ids()
        if self._label is not None:
            self._label.setText(f"Selected vertices: {len(ids):,}")
        if self._create_btn is not None:
            self._create_btn.setEnabled(len(ids) == 2)

    def _create_edge(self) -> None:
        layer = self._layer()
        ids = self._selected_vertex_ids()
        if layer is None or len(ids) != 2:
            return
        cmd = CreateCellComplexEdgeCmd(
            title="Create cell complex edge",
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
            vertex_id_a=ids[0],
            vertex_id_b=ids[1],
        )
        self.command_manager.do(cmd)
        self._update_context_ui()

    def left_button_release_hook(self, event: Event) -> None:
        layer = self._layer()
        if layer is None:
            return
        layer.handle_viewport_left_click(self.ctx, event, self.ctx.viewer.pick(prefer_cells=True))
        self._update_context_ui()

    def activate(self) -> None:
        return super().activate()

    def create_context_widget(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        layout = QHBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(2, 1, 2, 1)
        self._label = QLabel(w)
        layout.addWidget(self._label)
        self._create_btn = QPushButton("Create", w)
        self._create_btn.setIcon(QIcon(self.icon_path))
        self._create_btn.pressed.connect(self._create_edge)
        layout.addWidget(self._create_btn)
        self._update_context_ui()
        return w


@dataclass
class CellComplexAssociateTool(ModeTool):
    label: ClassVar = "cell_complex_associate"
    tooltip: ClassVar = "Associate CellComplex cell"
    icon_path: ClassVar = resource_path("inspect_tool.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    point_picking_active: bool = field(default=False, init=False)
    _layer_combo: QComboBox | None = field(default=None, init=False)
    _field_combo: QComboBox | None = field(default=None, init=False)
    _field_status: QLabel | None = field(default=None, init=False)
    _id_input: QLineEdit | None = field(default=None, init=False)
    _pick_btn: QPushButton | None = field(default=None, init=False)
    _accept_btn: QPushButton | None = field(default=None, init=False)
    _pick_cursor_active: bool = field(default=False, init=False)

    def _cell_layer(self) -> CellComplexLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, CellComplexLayer) else None

    def _selected_cell_id(self) -> str | None:
        layer = self._cell_layer()
        if layer is None:
            return None
        selected = list(layer.active_selection or ())
        return selected[0] if len(selected) == 1 else None

    def _point_layers(self) -> list[PointCloudLayer]:
        return [
            layer for layer in self.ctx.scene.layers.values()
            if isinstance(layer, PointCloudLayer)
        ]

    def _selected_point_layer(self) -> PointCloudLayer | None:
        if self._layer_combo is None:
            return None
        layer_id = self._layer_combo.currentData()
        layer = self.ctx.scene.layers.get(str(layer_id)) if layer_id is not None else None
        return layer if isinstance(layer, PointCloudLayer) else None

    def _selected_field(self) -> InstanceSegmentation | None:
        layer = self._selected_point_layer()
        if layer is None or self._field_combo is None:
            return None
        field_name = self._field_combo.currentData()
        fields = layer.data.get_fields(names=str(field_name), field_type=FieldType.INSTANCE)
        return fields[0] if fields and isinstance(fields[0], InstanceSegmentation) else None

    def _populate_fields(self) -> None:
        if self._field_combo is None or self._field_status is None:
            return
        self._field_combo.clear()
        layer = self._selected_point_layer()
        fields = (
            layer.data.get_fields(field_type=FieldType.INSTANCE)
            if layer is not None else []
        )
        for field in fields:
            self._field_combo.addItem(field.name, field.name)
        has_fields = bool(fields)
        self._field_combo.setVisible(has_fields)
        self._field_status.setVisible(not has_fields)
        self._update_accept_enabled()

    def _set_picking(self, active: bool) -> None:
        if self.point_picking_active == active:
            return
        self.point_picking_active = active
        if self._pick_btn is not None:
            if active:
                self._pick_btn.setText("Picking ...")
                self._pick_btn.setIcon(QIcon())
            else:
                self._pick_btn.setText("Pick instance")
                self._pick_btn.setIcon(QIcon(resource_path("inspect_tool.png")))
        if active and not self._pick_cursor_active:
            pixmap = QPixmap(resource_path("inspect_tool.png")).scaled(
                30, 30,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            QApplication.setOverrideCursor(QCursor(pixmap, 3, 3))
            self._pick_cursor_active = True
        elif not active and self._pick_cursor_active:
            QApplication.restoreOverrideCursor()
            self._pick_cursor_active = False

    def _update_accept_enabled(self) -> None:
        if self._accept_btn is None or self._id_input is None:
            return
        can_accept = (
            self._selected_cell_id() is not None
            and self._selected_point_layer() is not None
            and self._selected_field() is not None
            and self._id_input.text().strip() != ""
        )
        self._accept_btn.setEnabled(can_accept)

    def _pick_instance(self, event: Event) -> None:
        layer = self._selected_point_layer()
        field = self._selected_field()
        if layer is None or field is None:
            self._set_picking(False)
            return
        result = self.ctx.viewer.pick()
        if result.layer is not layer or result.element_idx is None:
            return
        idx = int(result.element_idx)
        if idx < 0 or idx >= field.data.shape[0]:
            return
        if self._id_input is not None:
            self._id_input.setText(str(int(field.data.reshape(-1)[idx])))
        self._set_picking(False)
        self._update_accept_enabled()

    def _accept(self) -> None:
        layer = self._cell_layer()
        pcd_layer = self._selected_point_layer()
        field = self._selected_field()
        cell_id = self._selected_cell_id()
        if layer is None or pcd_layer is None or field is None or cell_id is None or self._id_input is None:
            return
        try:
            instance_id = int(self._id_input.text().strip())
        except ValueError:
            return
        cmd = AddCellGeometryRefCmd(
            title="Associate cell geometry",
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
            cell_id=cell_id,
            ref=PointCloudRef(
                ref_id=pcd_layer.data.id,
                field_name=field.name,
                instance_id=instance_id,
            ),
        )
        self.command_manager.do(cmd)
        self.ctx.controller.deactivate_tool()

    def left_button_release_hook(self, event: Event) -> None:
        if self.point_picking_active:
            self._pick_instance(event)
            return
        layer = self._cell_layer()
        if layer is not None:
            layer.handle_viewport_left_click(self.ctx, event, self.ctx.viewer.pick(prefer_cells=True))
            self._update_accept_enabled()

    def activate(self) -> None:
        return super().activate()

    def deactivate(self) -> None:
        self._set_picking(False)
        return super().deactivate()

    def create_context_widget(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(3)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.addWidget(QLabel("Layer:", w))
        self._layer_combo = QComboBox(w)
        for layer in self._point_layers():
            self._layer_combo.addItem(layer.browser_name, layer.id)
        self._layer_combo.currentIndexChanged.connect(lambda _idx: self._populate_fields())
        row1.addWidget(self._layer_combo)
        row1.addWidget(QLabel("ID:", w))
        self._id_input = QLineEdit(w)
        self._id_input.setFixedWidth(80)
        self._id_input.textChanged.connect(lambda _txt: self._update_accept_enabled())
        row1.addWidget(self._id_input)
        self._pick_btn = QPushButton("Pick instance", w)
        self._pick_btn.setIcon(QIcon(resource_path("inspect_tool.png")))
        self._pick_btn.pressed.connect(lambda: self._set_picking(not self.point_picking_active))
        row1.addWidget(self._pick_btn)
        self._accept_btn = QPushButton("Accept", w)
        self._accept_btn.pressed.connect(self._accept)
        row1.addWidget(self._accept_btn)
        outer.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("Field:", w))
        self._field_combo = QComboBox(w)
        self._field_combo.currentIndexChanged.connect(lambda _idx: self._update_accept_enabled())
        row2.addWidget(self._field_combo)
        self._field_status = QLabel("Layer doesn't have instance fields.", w)
        self._field_status.setStyleSheet("QLabel { color: rgba(128,128,128,128); }")
        row2.addWidget(self._field_status)
        outer.addLayout(row2)

        self._populate_fields()
        return w
