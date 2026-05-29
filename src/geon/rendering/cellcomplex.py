from __future__ import annotations

import math
import time
import weakref
from typing import Any, Optional

import numpy as np
import vtk
from numpy.typing import NDArray
from PyQt6.QtCore import QTimer
from vtk.util import numpy_support as ns  # type: ignore

from geon.config import theme
from geon.data.cellcomplex import CellComplexData, EdgeCell, VertexCell

from .base import BaseLayer
from .layer_registry import layer_for


@layer_for(CellComplexData)
class CellComplexLayer(BaseLayer[CellComplexData]):
    layer_type_id = "cellcomplex"
    use_cell_picking_for_selection = True

    def __init__(self, data: CellComplexData, browser_name: str = "Cell Complex"):
        super().__init__(data)
        self.browser_name = browser_name

        self.size_mode: str = "screen"
        self.screen_size_px: float = 12.0
        self.world_size: float = 0.1
        self.edge_width: float = 1.0
        self.default_color: tuple[int, int, int] = theme.DEFAULT_SEGMENTATION_COLOR
        self.selection_color: tuple[int, int, int] = (255, 128, 0)
        self.face_opacity: float = 0.8
        self.active_semantic_attribute_by_dim: dict[int, str | None] = {
            0: None,
            1: None,
        }

        self._vertex_ids: list[str] = []
        self._vertex_id_to_index: dict[str, int] = {}
        self._edge_ids: list[str] = []
        self._edge_id_to_index: dict[str, int] = {}
        self._edge_render_ids: list[str] = []
        self._positions: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._active_selection: set[str] = set()

        self._face_poly: Optional[vtk.vtkPolyData] = None
        self._wire_poly: Optional[vtk.vtkPolyData] = None
        self._edge_poly: Optional[vtk.vtkPolyData] = None
        self._face_actor: Optional[vtk.vtkActor] = None
        self._wire_actor: Optional[vtk.vtkActor] = None
        self._edge_actor: Optional[vtk.vtkActor] = None
        self._camera_observer_id: Optional[int] = None
        self._observed_camera: Optional[vtk.vtkCamera] = None
        self._last_screen_scale_update: float = 0.0

        self._selection_phase: float = 0.0
        self._selection_timer = QTimer()
        self._selection_timer.setInterval(70)
        self._selection_timer.timeout.connect(self._update_selection_pulse)

    @property
    def active_selection(self) -> set[str] | None:
        return set(self._active_selection) if self._active_selection else None

    @active_selection.setter
    def active_selection(self, selection: set[str] | list[str] | tuple[str, ...] | None) -> None:
        known_ids = {cell.id for cell in self.data.get_cells()}
        self._active_selection = set(selection or ()) & known_ids
        self._sync_selection_timer()
        self.update()

    @property
    def browser_sel_descr(self) -> str | None:
        n = len(self._active_selection)
        if n == 0:
            return None
        return f"{n:,} cells"

    @property
    def vertex_count(self) -> int:
        return len(self.data.vertices)

    @property
    def edge_count(self) -> int:
        return len(self.data.edges)

    @property
    def selected_vertices(self) -> list[VertexCell]:
        selected = self._active_selection
        return [vertex for vertex in self.data.vertices if vertex.id in selected]

    @property
    def selected_edges(self) -> list[EdgeCell]:
        selected = self._active_selection
        return [edge for edge in self.data.edges if edge.id in selected]

    def selected_ids_by_dim(self, dim: int) -> list[str]:
        return [cell.id for cell in self.data.get_cells(dim) if cell.id in self._active_selection]

    def set_visual_settings(
        self,
        *,
        size_mode: str,
        screen_size_px: float,
        world_size: float,
        edge_width: float,
        default_color: tuple[int, int, int],
        selection_color: tuple[int, int, int] | None = None,
    ) -> None:
        self.size_mode = size_mode if size_mode in {"screen", "world"} else "world"
        self.screen_size_px = float(max(1.0, screen_size_px))
        self.world_size = float(max(1e-6, world_size))
        self.edge_width = float(max(1.0, edge_width))
        self.default_color = self._clamp_color(default_color)
        if selection_color is not None:
            self.selection_color = self._clamp_color(selection_color)
        self.update()

    def _build_vertex_cache(self) -> None:
        self._vertex_ids = [v.id for v in self.data.vertices]
        self._vertex_id_to_index = {vertex_id: i for i, vertex_id in enumerate(self._vertex_ids)}
        self._edge_ids = [e.id for e in self.data.edges]
        self._edge_id_to_index = {edge_id: i for i, edge_id in enumerate(self._edge_ids)}
        known_ids = set(self._vertex_ids) | set(self._edge_ids)
        self._active_selection &= known_ids
        if self.data.vertices:
            self._positions = np.asarray([v.position for v in self.data.vertices], dtype=np.float32)
        else:
            self._positions = np.empty((0, 3), dtype=np.float32)
        self._sync_selection_timer()

    def semantic_attribute_names(self, dim: int) -> list[str]:
        return sorted(self.data.semantic_attribute_schemas.get(dim, {}).keys())

    def set_active_semantic_attribute(self, dim: int, name: str | None) -> None:
        if name is not None and name not in self.semantic_attribute_names(dim):
            raise ValueError(f"Semantic attribute '{name}' is not available for dimension {dim}.")
        self.active_semantic_attribute_by_dim[dim] = name
        self.update()

    def _sync_active_semantic_attributes(self) -> None:
        for dim in range(2):
            active_name = self.active_semantic_attribute_by_dim.get(dim)
            if active_name is not None and active_name not in self.semantic_attribute_names(dim):
                self.active_semantic_attribute_by_dim[dim] = None

    @staticmethod
    def _clamp_color(color: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
        return tuple(int(max(0, min(255, c))) for c in color)  # type: ignore[return-value]

    def _base_color(self, cell: VertexCell | EdgeCell) -> tuple[int, int, int]:
        attr_name = self.active_semantic_attribute_by_dim.get(cell.dim)
        if attr_name is None:
            return self.default_color
        schema = self.data.semantic_attribute_schemas.get(cell.dim, {}).get(attr_name)
        if schema is None:
            return self.default_color
        class_id = cell.semantic_attributes.get(attr_name)
        if class_id is None:
            return self.default_color
        try:
            sem_class = schema.by_id(int(class_id))
        except IndexError:
            return self.default_color
        return self._clamp_color(sem_class.color)

    def _pulse_color(self) -> tuple[int, int, int]:
        pulse = 0.55 + 0.45 * math.sin(self._selection_phase)
        base = np.asarray(self.selection_color, dtype=np.float32)
        white = np.asarray((255, 230, 180), dtype=np.float32)
        color = base * (1.0 - pulse) + white * pulse
        return self._clamp_color(color.astype(np.int32).tolist())

    def _vertex_colors(self) -> NDArray[np.uint8]:
        pulse = self._pulse_color()
        colors: list[tuple[int, int, int]] = []
        selected = self._active_selection
        for vertex in self.data.vertices:
            colors.append(pulse if vertex.id in selected else self._base_color(vertex))
        return np.asarray(colors, dtype=np.uint8).reshape((-1, 3))

    def _edge_color(self, edge: EdgeCell) -> tuple[int, int, int]:
        if edge.id in self._active_selection:
            return self._pulse_color()
        return self._base_color(edge)

    def _sync_selection_timer(self) -> None:
        if self._active_selection and self._renderer is not None:
            if not self._selection_timer.isActive():
                self._selection_timer.start()
        elif self._selection_timer.isActive():
            self._selection_timer.stop()

    def _update_selection_pulse(self) -> None:
        if not self._active_selection:
            self._sync_selection_timer()
            return
        self._selection_phase = (self._selection_phase + 0.35) % (2.0 * math.pi)
        self._update_cube_polys()
        self._update_edge_poly()
        if self._renderer is not None:
            self._renderer.GetRenderWindow().Render()

    def _size_array(self) -> NDArray[np.float32]:
        if self._positions.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        if self.size_mode == "world" or self.renderer is None:
            return np.full((self._positions.shape[0],), self.world_size, dtype=np.float32)

        renderer = self.renderer
        camera = renderer.GetActiveCamera()
        width, height = renderer.GetRenderWindow().GetSize()
        viewport_h = max(1, int(height))

        if camera.GetParallelProjection():
            world_per_px = 2.0 * float(camera.GetParallelScale()) / float(viewport_h)
            size = self.screen_size_px * world_per_px
            return np.full((self._positions.shape[0],), max(size, 1e-6), dtype=np.float32)

        fov_rad = math.radians(float(camera.GetViewAngle()))
        cam_pos = np.asarray(camera.GetPosition(), dtype=np.float32)
        distances = np.linalg.norm(self._positions - cam_pos[None, :], axis=1)
        world_per_px = 2.0 * distances * math.tan(fov_rad / 2.0) / float(viewport_h)
        return np.maximum(self.screen_size_px * world_per_px, 1e-6).astype(np.float32)

    def _cube_geometry(self) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        n = self._positions.shape[0]
        if n == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        half_sizes = (self._size_array() * 0.5).astype(np.float32)
        offsets_unit = np.asarray(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float32,
        )
        points = (
            self._positions[:, None, :]
            + offsets_unit[None, :, :] * half_sizes[:, None, None]
        ).reshape((-1, 3))
        colors = np.repeat(self._vertex_colors(), 8, axis=0)
        return points.astype(np.float32, copy=False), colors.astype(np.uint8, copy=False)

    def _set_point_colors(self, poly: vtk.vtkPolyData, colors: NDArray[np.uint8]) -> None:
        vtk_colors = ns.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("colors")
        vtk_colors.SetNumberOfComponents(3)
        poly.GetPointData().SetScalars(vtk_colors)

    def _update_cube_polys(self) -> None:
        if self._face_poly is None or self._wire_poly is None:
            return
        cube_points, colors = self._cube_geometry()

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(cube_points, deep=True))

        faces = vtk.vtkCellArray()
        wires = vtk.vtkCellArray()
        face_quads = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        wire_edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for i in range(self._positions.shape[0]):
            base = i * 8
            for quad in face_quads:
                cell = vtk.vtkQuad()
                for j, pid in enumerate(quad):
                    cell.GetPointIds().SetId(j, base + pid)
                faces.InsertNextCell(cell)
            for a, b in wire_edges:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, base + a)
                line.GetPointIds().SetId(1, base + b)
                wires.InsertNextCell(line)

        self._face_poly.SetPoints(vtk_points)
        self._face_poly.SetPolys(faces)
        self._set_point_colors(self._face_poly, colors)
        self._face_poly.Modified()

        wire_points = vtk.vtkPoints()
        wire_points.SetData(ns.numpy_to_vtk(cube_points, deep=True))
        self._wire_poly.SetPoints(wire_points)
        self._wire_poly.SetLines(wires)
        self._set_point_colors(self._wire_poly, colors)
        self._wire_poly.Modified()

    def _update_edge_poly(self) -> None:
        if self._edge_poly is None:
            return
        points = vtk.vtkPoints()
        points.SetData(ns.numpy_to_vtk(self._positions, deep=True))
        lines = vtk.vtkCellArray()
        edge_colors: list[tuple[int, int, int]] = []
        self._edge_render_ids = []
        for edge in self.data.edges:
            if len(edge.boundary) != 2:
                continue
            a = self._vertex_id_to_index.get(edge.boundary[0])
            b = self._vertex_id_to_index.get(edge.boundary[1])
            if a is None or b is None:
                continue
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(a))
            line.GetPointIds().SetId(1, int(b))
            lines.InsertNextCell(line)
            self._edge_render_ids.append(edge.id)
            edge_colors.append(self._edge_color(edge))
        self._edge_poly.SetPoints(points)
        self._edge_poly.SetLines(lines)
        if edge_colors:
            vtk_colors = ns.numpy_to_vtk(
                np.asarray(edge_colors, dtype=np.uint8),
                deep=True,
                array_type=vtk.VTK_UNSIGNED_CHAR,
            )
            vtk_colors.SetName("colors")
            vtk_colors.SetNumberOfComponents(3)
            self._edge_poly.GetCellData().SetScalars(vtk_colors)
        else:
            self._edge_poly.GetCellData().SetScalars(None)
        self._edge_poly.Modified()

    def _make_cube_actor(self, poly: vtk.vtkPolyData, wireframe: bool) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if wireframe:
            prop.SetRepresentationToWireframe()
            prop.SetOpacity(1.0)
            prop.SetLineWidth(self.edge_width)
        else:
            prop.SetRepresentationToSurface()
            prop.SetOpacity(self.face_opacity)
        return actor

    def _build_pipeline(
        self,
        renderer: vtk.vtkRenderer,
        out_actors: list[vtk.vtkProp],
    ) -> None:
        self._build_vertex_cache()
        self._face_poly = vtk.vtkPolyData()
        self._wire_poly = vtk.vtkPolyData()
        self._edge_poly = vtk.vtkPolyData()
        self._update_cube_polys()
        self._update_edge_poly()

        self._face_actor = self._make_cube_actor(self._face_poly, wireframe=False)
        self._wire_actor = self._make_cube_actor(self._wire_poly, wireframe=True)

        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputData(self._edge_poly)
        edge_mapper.SetScalarModeToUseCellData()
        edge_mapper.SetColorModeToDirectScalars()
        self._edge_actor = vtk.vtkActor()
        self._edge_actor.SetMapper(edge_mapper)
        self._edge_actor.GetProperty().SetColor(*(c / 255.0 for c in self.default_color))
        self._edge_actor.GetProperty().SetLineWidth(self.edge_width)

        out_actors.extend([self._face_actor, self._wire_actor, self._edge_actor])

        camera = renderer.GetActiveCamera()
        self._observed_camera = camera
        self._camera_observer_id = camera.AddObserver(
            vtk.vtkCommand.ModifiedEvent,
            lambda _obj, _evt: self._on_camera_modified(),
        )
        self._sync_selection_timer()

    def _on_camera_modified(self) -> None:
        if self.size_mode != "screen":
            return
        now = time.perf_counter()
        if now - self._last_screen_scale_update < 0.03:
            return
        self._last_screen_scale_update = now
        self._update_cube_polys()

    def update(self) -> None:
        self._sync_active_semantic_attributes()
        self._build_vertex_cache()
        self._update_cube_polys()
        self._update_edge_poly()
        if self._wire_actor is not None:
            self._wire_actor.GetProperty().SetLineWidth(self.edge_width)
        if self._edge_actor is not None:
            self._edge_actor.GetProperty().SetLineWidth(self.edge_width)
            self._edge_actor.GetProperty().SetColor(*(c / 255.0 for c in self.default_color))

    def prefers_cell_pick(self, prop: vtk.vtkProp | None) -> bool:
        return prop in {None, self._face_actor, self._wire_actor, self._edge_actor}

    def _vertex_index_from_pick(
        self,
        sub_id: int,
        prop: vtk.vtkProp | None,
        association: str | None = None,
    ) -> int:
        if sub_id < 0:
            return -1
        if prop is self._face_actor and association == "cell":
            return int(sub_id // 6)
        if prop is self._wire_actor and association == "cell":
            return int(sub_id // 12)
        if prop is self._face_actor or prop is self._wire_actor:
            return int(sub_id // 8)
        return int(sub_id)

    def _edge_index_from_pick(self, sub_id: int) -> int:
        if sub_id < 0 or sub_id >= len(self._edge_render_ids):
            return -1
        edge_id = self._edge_render_ids[sub_id]
        return self._edge_id_to_index.get(edge_id, -1)

    def cell_id_from_pick(
        self,
        sub_id: int,
        prop: vtk.vtkProp | None,
        association: str | None = None,
    ) -> str | None:
        if prop is self._edge_actor and association == "cell":
            if 0 <= sub_id < len(self._edge_render_ids):
                return self._edge_render_ids[sub_id]
            return None
        vertex_index = self._vertex_index_from_pick(sub_id, prop, association)
        if 0 <= vertex_index < len(self._vertex_ids):
            return self._vertex_ids[vertex_index]
        return None

    def data_index_from_picked_id(self, sub_id: int) -> int:
        return self._vertex_index_from_pick(sub_id, None)

    def data_index_from_pick(
        self,
        sub_id: int,
        prop: vtk.vtkProp | None,
        association: str | None = None,
    ) -> int:
        if prop is self._edge_actor and association == "cell":
            return self._edge_index_from_pick(sub_id)
        return self._vertex_index_from_pick(sub_id, prop, association)

    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float, float, float]:
        vertex_index = self._vertex_index_from_pick(sub_id, None)
        if vertex_index < 0 or vertex_index >= self._positions.shape[0]:
            raise IndexError(f"Picked vertex index out of range: {sub_id}")
        p = self._positions[int(vertex_index)]
        return float(p[0]), float(p[1]), float(p[2])

    def world_xyz_from_pick(
        self,
        sub_id: int,
        prop: vtk.vtkProp | None,
        association: str | None = None,
    ) -> tuple[float, float, float]:
        if prop is self._edge_actor and association == "cell":
            edge_id = self.cell_id_from_pick(sub_id, prop, association)
            edge = next((e for e in self.data.edges if e.id == edge_id), None)
            if edge is not None:
                idx = [self._vertex_id_to_index.get(v_id) for v_id in edge.boundary]
                if all(i is not None for i in idx):
                    pts = self._positions[np.asarray(idx, dtype=np.int32)]
                    midpoint = pts.mean(axis=0)
                    return float(midpoint[0]), float(midpoint[1]), float(midpoint[2])
        return self.world_xyz_from_picked_id(self._vertex_index_from_pick(sub_id, prop, association))

    def _selection_with_pick(self, event: Any, picked_id: str) -> set[str]:
        old = set(self._active_selection)
        if event.ctrl:
            old.discard(picked_id)
            return old
        if event.shift:
            old.add(picked_id)
            return old
        return {picked_id}

    def handle_viewport_left_click(self, ctx: Any, event: Any, pick_result: Any) -> bool:
        if pick_result.layer is not self or pick_result.prop is None or pick_result.element_idx is None:
            return False
        raw_element_idx = getattr(pick_result, "raw_element_idx", pick_result.element_idx)
        picked_id = self.cell_id_from_pick(
            int(raw_element_idx),
            pick_result.prop,
            pick_result.association,
        )
        if picked_id is None:
            return False

        from geon.tools.cellcomplex import SelectCellComplexCellsCmd

        new_selection = self._selection_with_pick(event, picked_id)
        cmd = SelectCellComplexCellsCmd(
            title="Select cell complex cells",
            layer_ref=weakref.ref(self),
            ctx_ref=weakref.ref(ctx),
            selection_new=new_selection,
            selection_old=set(self._active_selection),
        )
        ctx.controller.command_manager.do(cmd)
        return True

    def viewport_context_actions(self, ctx: Any, pick_result: Any) -> list[Any]:
        from PyQt6.QtGui import QAction
        from geon.tools.cellcomplex import DeleteCellComplexCellsCmd
        from geon.ui.cellcomplex_dialogs import CellComplexAttributesDialog

        actions: list[QAction] = []
        vertex_ids = set(self.selected_ids_by_dim(0))
        edge_ids = set(self.selected_ids_by_dim(1))

        if vertex_ids:
            action = QAction(f"Delete selected {len(vertex_ids):,} vertices", None)
            action.triggered.connect(
                lambda _checked=False, ids=vertex_ids: ctx.controller.command_manager.do(
                    DeleteCellComplexCellsCmd(
                        title="Delete cell complex vertices",
                        layer_ref=weakref.ref(self),
                        ctx_ref=weakref.ref(ctx),
                        vertex_ids=set(ids),
                        edge_ids=set(),
                    )
                )
            )
            actions.append(action)

        if edge_ids:
            action = QAction(f"Delete selected {len(edge_ids):,} edges", None)
            action.triggered.connect(
                lambda _checked=False, ids=edge_ids: ctx.controller.command_manager.do(
                    DeleteCellComplexCellsCmd(
                        title="Delete cell complex edges",
                        layer_ref=weakref.ref(self),
                        ctx_ref=weakref.ref(ctx),
                        vertex_ids=set(),
                        edge_ids=set(ids),
                    )
                )
            )
            actions.append(action)

        selected_ids = list(self._active_selection)
        if len(selected_ids) == 1:
            cell = self.data.get_cell_by_id(selected_ids[0])
            action = QAction("Edit attributes...", None)
            action.triggered.connect(
                lambda _checked=False, c=cell: (
                    CellComplexAttributesDialog(self, c, ctx, ctx.viewer).exec()
                    if c is not None else None
                )
            )
            actions.append(action)

        return actions

    def detach(self) -> None:
        self._selection_timer.stop()
        if self._camera_observer_id is not None and self._observed_camera is not None:
            self._observed_camera.RemoveObserver(self._camera_observer_id)
        self._camera_observer_id = None
        self._observed_camera = None
        super().detach()
