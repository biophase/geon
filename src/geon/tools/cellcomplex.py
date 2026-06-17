from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional
import weakref

import numpy as np

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QCursor, QIcon, QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from geon.data.cellcomplex import EdgeCell, PointCloudRef
from geon.data.pointcloud import FieldType, InstanceSegmentation, SemanticClass, SemanticSchema
from geon.rendering.cellcomplex import CellComplexLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.util.resources import resource_path

from .base import Event, ModeTool
from .command_manager import Command
from .tool_context import ToolContext


def _refresh_context(ctx: ToolContext, layer: CellComplexLayer) -> None:
    layer.data.normalize_and_validate()
    layer.update()
    ctx.controller.layer_internal_sel_changed.emit(layer)
    ctx.controller.scene_tree_request_change.emit()
    mark_modified = getattr(ctx.viewer.window(), "_mark_active_doc_modified", None)
    if callable(mark_modified):
        mark_modified()
    ctx.viewer.rerender()


def cell_complex_annotation_target(
    layer: CellComplexLayer,
) -> tuple[int, str, SemanticSchema] | None:
    selected = set(layer.active_selection or ())
    if not selected:
        return None
    dims: set[int] = set()
    for cell_id in selected:
        cell = layer.data.get_cell_by_id(cell_id)
        if cell is None:
            continue
        dims.add(cell.dim)
    if len(dims) != 1:
        return None
    dim = next(iter(dims))
    attr_name = layer.active_semantic_attribute_by_dim.get(dim)
    if attr_name is None:
        return None
    schema = layer.data.semantic_attribute_schemas.get(dim, {}).get(attr_name)
    if schema is None:
        return None
    return dim, attr_name, schema


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
class AnnotateCellComplexSemanticCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    cell_ids: set[str]
    dim: int
    attr_name: str
    semantic_class: SemanticClass
    _old_values: dict[str, int] = field(default_factory=dict, init=False)

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        self._old_values = {}
        for cell_id in self.cell_ids:
            cell = layer.data.get_cell_by_id(cell_id)
            if cell is None or cell.dim != self.dim:
                continue
            if self.attr_name not in cell.semantic_attributes:
                continue
            self._old_values[cell_id] = int(cell.semantic_attributes[self.attr_name])
            cell.semantic_attributes[self.attr_name] = int(self.semantic_class.id)
        _refresh_context(ctx, layer)

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        for cell_id, old_value in self._old_values.items():
            cell = layer.data.get_cell_by_id(cell_id)
            if cell is None:
                continue
            if self.attr_name in cell.semantic_attributes:
                cell.semantic_attributes[self.attr_name] = int(old_value)
        _refresh_context(ctx, layer)


@dataclass
class TranslateCellComplexVerticesCmd(Command):
    layer_ref: weakref.ReferenceType[CellComplexLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    positions_old: dict[str, tuple[float, float, float]]
    positions_new: dict[str, tuple[float, float, float]]

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.data.set_vertex_positions(self.positions_new)
        _refresh_context(ctx, layer)

    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.data.set_vertex_positions(self.positions_old)
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
class CellComplexMoveNodesTool(ModeTool):
    label: ClassVar = "cell_complex_move_nodes"
    tooltip: ClassVar = "Move CellComplex nodes"
    icon_path: ClassVar = resource_path("inspect_tool.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    _drag_handle: str | None = field(default=None, init=False)
    _drag_origin: tuple[float, float, float] | None = field(default=None, init=False)
    _drag_start_point: np.ndarray | None = field(default=None, init=False)
    _positions_old: dict[str, tuple[float, float, float]] = field(default_factory=dict, init=False)

    def _layer(self) -> CellComplexLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, CellComplexLayer) else None

    def _selected_vertex_positions(self, layer: CellComplexLayer) -> dict[str, tuple[float, float, float]]:
        return {
            vertex.id: tuple(float(v) for v in vertex.position)
            for vertex in layer.selected_vertices
        }

    def _display_ray(self, event: Event) -> tuple[np.ndarray, np.ndarray]:
        renderer = self.ctx.viewer._renderer
        x, y = event.pos

        def display_to_world(z: float) -> np.ndarray:
            renderer.SetDisplayPoint(float(x), float(y), float(z))
            renderer.DisplayToWorld()
            wx, wy, wz, ww = renderer.GetWorldPoint()
            if abs(float(ww)) < 1e-12:
                return np.asarray([wx, wy, wz], dtype=np.float64)
            return np.asarray([wx / ww, wy / ww, wz / ww], dtype=np.float64)

        p0 = display_to_world(0.0)
        p1 = display_to_world(1.0)
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return p0, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return p0, direction / norm

    @staticmethod
    def _axis_for_handle(handle: str) -> np.ndarray:
        return {
            "axis_x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            "axis_y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "axis_z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        }[handle]

    @staticmethod
    def _plane_normal_for_handle(handle: str) -> np.ndarray:
        return {
            "plane_xy": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "plane_yz": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            "plane_xz": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        }[handle]

    def _camera_plane_normal(self) -> np.ndarray:
        camera = self.ctx.viewer._renderer.GetActiveCamera()
        n = np.asarray(camera.GetDirectionOfProjection(), dtype=np.float64)
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 1e-12 else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    def _point_on_axis(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        origin: np.ndarray,
        axis: np.ndarray,
    ) -> np.ndarray | None:
        matrix = np.column_stack((axis, -ray_dir))
        rhs = ray_origin - origin
        try:
            params, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return origin + axis * float(params[0])

    def _point_on_plane(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray | None:
        denom = float(np.dot(ray_dir, normal))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(origin - ray_origin, normal) / denom)
        return ray_origin + ray_dir * t

    def _constraint_point(self, event: Event, handle: str, origin: tuple[float, float, float]) -> np.ndarray | None:
        ray_origin, ray_dir = self._display_ray(event)
        origin_np = np.asarray(origin, dtype=np.float64)
        if handle.startswith("axis_"):
            return self._point_on_axis(ray_origin, ray_dir, origin_np, self._axis_for_handle(handle))
        if handle.startswith("plane_"):
            return self._point_on_plane(ray_origin, ray_dir, origin_np, self._plane_normal_for_handle(handle))
        return self._point_on_plane(ray_origin, ray_dir, origin_np, self._camera_plane_normal())

    def _pick_gizmo_handle(self) -> tuple[CellComplexLayer | None, str | None]:
        layer = self._layer()
        if layer is None:
            return None, None
        pick = self.ctx.viewer.pick()
        if pick.layer is not layer:
            return layer, None
        return layer, layer.gizmo_handle_from_prop(pick.prop)

    def activate(self) -> None:
        super().activate()
        layer = self._layer()
        if layer is not None:
            layer.set_node_gizmo_enabled(True)
        self.ctx.viewer.set_camera_enabled(False)
        self.ctx.viewer.rerender()

    def deactivate(self) -> None:
        layer = self._layer()
        if layer is not None:
            layer.set_node_gizmo_enabled(False)
            layer.set_gizmo_hover_handle(None)
        self.ctx.viewer.set_camera_enabled(True)
        self.ctx.viewer.rerender()
        return super().deactivate()

    def left_button_press_hook(self, event: Event) -> None:
        layer, handle = self._pick_gizmo_handle()
        if layer is None or handle is None or layer.node_gizmo_origin is None:
            return
        positions = self._selected_vertex_positions(layer)
        if not positions:
            return
        start_point = self._constraint_point(event, handle, layer.node_gizmo_origin)
        if start_point is None:
            return
        self._drag_handle = handle
        self._drag_origin = layer.node_gizmo_origin
        self._drag_start_point = start_point
        self._positions_old = positions

    def mouse_move_event_hook(self, event: Event) -> None:
        layer = self._layer()
        if layer is None:
            return
        if self._drag_handle is None:
            _layer, handle = self._pick_gizmo_handle()
            layer.set_gizmo_hover_handle(handle)
            self.ctx.viewer.rerender()
            return
        if self._drag_origin is None or self._drag_start_point is None:
            return
        current = self._constraint_point(event, self._drag_handle, self._drag_origin)
        if current is None:
            return
        delta = current - self._drag_start_point
        preview = {
            vertex_id: (
                old_pos[0] + float(delta[0]),
                old_pos[1] + float(delta[1]),
                old_pos[2] + float(delta[2]),
            )
            for vertex_id, old_pos in self._positions_old.items()
        }
        layer.data.set_vertex_positions(preview)
        layer.update()
        self.ctx.viewer.rerender()

    def left_button_release_hook(self, event: Event) -> None:
        layer = self._layer()
        if layer is None:
            return
        if self._drag_handle is None:
            layer.handle_viewport_left_click(self.ctx, event, self.ctx.viewer.pick(prefer_cells=True))
            return
        positions_new = self._selected_vertex_positions(layer)
        positions_old = dict(self._positions_old)
        layer.data.set_vertex_positions(positions_old)
        layer.update()
        self._drag_handle = None
        self._drag_origin = None
        self._drag_start_point = None
        self._positions_old = {}

        if positions_new == positions_old:
            self.ctx.viewer.rerender()
            return
        cmd = TranslateCellComplexVerticesCmd(
            title="Translate cell complex nodes",
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
            positions_old=positions_old,
            positions_new=positions_new,
        )
        self.command_manager.do(cmd)

    def create_context_widget(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.addWidget(QLabel("Drag gizmo handles to move selected nodes.", w))
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


@dataclass
class CellComplexAnnotateTool(ModeTool):
    label: ClassVar = "cell_complex_annotate"
    tooltip: ClassVar = "Annotate CellComplex semantics"
    icon_path: ClassVar = resource_path("annotate.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    choice_sem_class: Optional[SemanticClass] = None
    _class_input: QLineEdit | None = field(default=None, init=False)
    _accept_btn: QPushButton | None = field(default=None, init=False)
    _completer: QCompleter | None = field(default=None, init=False)
    _class_model: QStandardItemModel | None = field(default=None, init=False)
    _classes: list[SemanticClass] = field(default_factory=list, init=False)

    def _layer(self) -> CellComplexLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, CellComplexLayer) else None

    def _target(self) -> tuple[int, str, SemanticSchema] | None:
        layer = self._layer()
        if layer is None:
            return None
        return cell_complex_annotation_target(layer)

    @staticmethod
    def _build_sem_class_model(classes: list[SemanticClass]) -> QStandardItemModel:
        model = QStandardItemModel()
        for sem_class in classes:
            item = QStandardItem(sem_class.name)
            r, g, b = sem_class.color
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(r, g, b))
            item.setIcon(QIcon(swatch))
            model.appendRow(item)
        return model

    def _update_accept_enabled(self) -> None:
        if self._accept_btn is None:
            return
        self._accept_btn.setEnabled(self._target() is not None and self.choice_sem_class is not None)

    def _set_class_from_text(self, text: str) -> None:
        name = text.strip()
        self.choice_sem_class = next((cls for cls in self._classes if cls.name == name), None)
        self._update_accept_enabled()

    def _show_sem_class_popup(self) -> None:
        if self._class_input is None or self._completer is None:
            return
        self._class_input.setFocus(Qt.FocusReason.OtherFocusReason)
        self._completer.setCompletionPrefix("")
        self._completer.complete(self._class_input.rect())

    def _accept(self) -> None:
        layer = self._layer()
        target = self._target()
        if layer is None or target is None or self.choice_sem_class is None:
            return
        dim, attr_name, _schema = target
        cell_ids = {
            cell_id
            for cell_id in set(layer.active_selection or ())
            if (layer.data.get_cell_by_id(cell_id) is not None
                and layer.data.get_cell_by_id(cell_id).dim == dim)
        }
        if not cell_ids:
            return
        cmd = AnnotateCellComplexSemanticCmd(
            title="Annotate cell complex semantics",
            layer_ref=weakref.ref(layer),
            ctx_ref=weakref.ref(self.ctx),
            cell_ids=cell_ids,
            dim=dim,
            attr_name=attr_name,
            semantic_class=self.choice_sem_class,
        )
        self.command_manager.do(cmd)
        self.ctx.controller.deactivate_tool()

    def left_button_release_hook(self, event: Event) -> None:
        layer = self._layer()
        if layer is not None:
            layer.handle_viewport_left_click(self.ctx, event, self.ctx.viewer.pick(prefer_cells=True))
        self._update_accept_enabled()

    def activate(self) -> None:
        return super().activate()

    def create_context_widget(self, parent: QWidget) -> QWidget:
        target = self._target()
        if target is None:
            w = QWidget(parent)
            layout = QHBoxLayout(w)
            layout.setContentsMargins(2, 1, 2, 1)
            label = QLabel("Select cells from one dimension and activate a semantic attribute.", w)
            label.setStyleSheet("QLabel { color: rgba(128,128,128,160); }")
            layout.addWidget(label)
            return w

        dim, attr_name, schema = target
        self._classes = list(schema.semantic_classes)

        w = QWidget(parent)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(4)

        outer.addWidget(QLabel(f"Dim {dim}: {attr_name}", w))
        outer.addWidget(QLabel("Class:", w))

        self._class_input = QLineEdit(w)
        self._class_input.setPlaceholderText("Enter class ...")
        self._class_model = self._build_sem_class_model(self._classes)
        self._completer = QCompleter(self._class_model, self._class_input)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        popup = self._completer.popup()
        if popup is not None:
            popup = popup if isinstance(popup, QListView) else QListView(self._class_input)
            popup.setUniformItemSizes(True)
            popup.setIconSize(QSize(12, 12))
            popup.setMinimumWidth(max(160, popup.sizeHintForColumn(0) + 24))
            self._completer.setPopup(popup)
        self._class_input.setCompleter(self._completer)
        self._class_input.textEdited.connect(self._set_class_from_text)
        self._completer.activated.connect(self._set_class_from_text)
        outer.addWidget(self._class_input)

        dropdown = QPushButton(w)
        dropdown.setFixedSize(20, 20)
        dropdown.setText("▼")
        dropdown.setStyleSheet("font-size: 18Â§px;")
        dropdown.setToolTip("Select semantic class")
        dropdown.clicked.connect(self._show_sem_class_popup)
        outer.addWidget(dropdown)

        self._accept_btn = QPushButton("Accept", w)
        self._accept_btn.pressed.connect(self._accept)
        outer.addWidget(self._accept_btn)
        self._update_accept_enabled()
        return w
