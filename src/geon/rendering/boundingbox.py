from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import vtk
from numpy.typing import NDArray
from PyQt6.QtCore import QTimer
from vtk.util import numpy_support as ns  # type: ignore

from geon.config import theme
from geon.data.boundingbox import FACE_NAMES, BoundingBox, BoundingBoxData

from .base import BaseLayer
from .layer_registry import layer_for


WIRE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@layer_for(BoundingBoxData)
class BoundingBoxLayer(BaseLayer[BoundingBoxData]):
    layer_type_id = "boundingbox"
    use_cell_picking_for_selection = True

    def __init__(self, data: BoundingBoxData, browser_name: str = "Bounding Boxes"):
        super().__init__(data)
        self.browser_name = browser_name
        self.default_color: tuple[int, int, int] = theme.DEFAULT_SEGMENTATION_COLOR
        self.selection_color: tuple[int, int, int] = theme.SELECTION_COLOR
        self.face_opacity: float = 0.22
        self.line_width: float = 2.0
        self._active_selection: set[str] = set()
        self._selected_face: tuple[str, str] | None = None
        self._face_render_refs: list[tuple[str, str]] = []
        self._wire_render_refs: list[str] = []

        self._face_poly: Optional[vtk.vtkPolyData] = None
        self._wire_poly: Optional[vtk.vtkPolyData] = None
        self._face_actor: Optional[vtk.vtkActor] = None
        self._wire_actor: Optional[vtk.vtkActor] = None
        self._arrow_actor: Optional[vtk.vtkActor] = None
        self._arrow_box_face: tuple[str, str] | None = None

        self._selection_phase: float = 0.0
        self._selection_timer = QTimer()
        self._selection_timer.setInterval(70)
        self._selection_timer.timeout.connect(self._update_selection_pulse)

    @staticmethod
    def _clamp_color(color: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
        return tuple(int(max(0, min(255, c))) for c in color)  # type: ignore[return-value]

    def set_visual_settings(
        self,
        *,
        default_color: tuple[int, int, int],
        selection_color: tuple[int, int, int],
    ) -> None:
        self.default_color = self._clamp_color(default_color)
        self.selection_color = self._clamp_color(selection_color)
        self.update()

    @property
    def active_selection(self) -> set[str] | None:
        return set(self._active_selection) if self._active_selection else None

    @active_selection.setter
    def active_selection(self, selection: set[str] | list[str] | tuple[str, ...] | None) -> None:
        known_ids = {box.id for box in self.data.boxes}
        self._active_selection = set(selection or ()) & known_ids
        if self._selected_face is not None and self._selected_face[0] not in self._active_selection:
            self._selected_face = None
        self._sync_selection_timer()
        self.update()

    @property
    def selected_face(self) -> tuple[str, str] | None:
        return self._selected_face

    @selected_face.setter
    def selected_face(self, value: tuple[str, str] | None) -> None:
        if value is not None:
            box_id, face = value
            if self.data.get_box(box_id) is None or face not in FACE_NAMES:
                value = None
        self._selected_face = value
        if value is not None:
            self._active_selection = {value[0]}
        self._sync_selection_timer()
        self.update()

    @property
    def browser_sel_descr(self) -> str | None:
        n = len(self._active_selection)
        if n == 0:
            return None
        return f"{n:,} boxes"

    @property
    def box_count(self) -> int:
        return self.data.box_count

    def _sync_selection_timer(self) -> None:
        has_selection = bool(self._active_selection or self._selected_face)
        if has_selection and self._renderer is not None:
            if not self._selection_timer.isActive():
                self._selection_timer.start()
        elif self._selection_timer.isActive():
            self._selection_timer.stop()

    def _pulse_color(self) -> tuple[int, int, int]:
        pulse = 0.55 + 0.45 * math.sin(self._selection_phase)
        base = np.asarray(self.selection_color, dtype=np.float32)
        white = np.asarray((255, 230, 180), dtype=np.float32)
        color = base * (1.0 - pulse) + white * pulse
        return self._clamp_color(color.astype(np.int32).tolist())

    def _semantic_color(self, box: BoundingBox) -> tuple[int, int, int]:
        if self.data.schema is None or box.semantic_id is None:
            return self.default_color
        try:
            return self._clamp_color(self.data.schema.by_id(int(box.semantic_id)).color)
        except IndexError:
            return self.default_color

    def _box_color(self, box: BoundingBox, face: str | None = None) -> tuple[int, int, int]:
        if face is not None and self._selected_face == (box.id, face):
            return self._pulse_color()
        if box.id in self._active_selection:
            return self._pulse_color()
        return self._semantic_color(box)

    def _geometry(self) -> tuple[NDArray[np.float32], list[tuple[int, int, int, int]], list[tuple[int, int]], NDArray[np.uint8], NDArray[np.uint8]]:
        points: list[NDArray[np.float64]] = []
        faces: list[tuple[int, int, int, int]] = []
        wires: list[tuple[int, int]] = []
        face_colors: list[tuple[int, int, int]] = []
        wire_colors: list[tuple[int, int, int]] = []
        self._face_render_refs = []
        self._wire_render_refs = []
        for box in self.data.boxes:
            corners = box.corners()
            base = len(points) * 8
            points.append(corners)
            quads = box.face_quads()
            for face in FACE_NAMES:
                faces.append(tuple(base + idx for idx in quads[face]))
                face_colors.append(self._box_color(box, face))
                self._face_render_refs.append((box.id, face))
            for a, b in WIRE_EDGES:
                wires.append((base + a, base + b))
                wire_colors.append(self._box_color(box))
                self._wire_render_refs.append(box.id)
        if points:
            pts = np.concatenate(points, axis=0).astype(np.float32)
        else:
            pts = np.empty((0, 3), dtype=np.float32)
        return (
            pts,
            faces,
            wires,
            np.asarray(face_colors, dtype=np.uint8).reshape((-1, 3)),
            np.asarray(wire_colors, dtype=np.uint8).reshape((-1, 3)),
        )

    @staticmethod
    def _set_cell_colors(poly: vtk.vtkPolyData, colors: NDArray[np.uint8]) -> None:
        if colors.size == 0:
            poly.GetCellData().SetScalars(None)
            return
        vtk_colors = ns.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("colors")
        vtk_colors.SetNumberOfComponents(3)
        poly.GetCellData().SetScalars(vtk_colors)

    def _update_polys(self) -> None:
        if self._face_poly is None or self._wire_poly is None:
            return
        points_np, faces, wires, face_colors, wire_colors = self._geometry()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(points_np, deep=True))

        face_cells = vtk.vtkCellArray()
        for quad_ids in faces:
            quad = vtk.vtkQuad()
            for i, pid in enumerate(quad_ids):
                quad.GetPointIds().SetId(i, int(pid))
            face_cells.InsertNextCell(quad)
        self._face_poly.SetPoints(vtk_points)
        self._face_poly.SetPolys(face_cells)
        self._set_cell_colors(self._face_poly, face_colors)
        self._face_poly.Modified()

        wire_points = vtk.vtkPoints()
        wire_points.SetData(ns.numpy_to_vtk(points_np, deep=True))
        wire_cells = vtk.vtkCellArray()
        for a, b in wires:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(a))
            line.GetPointIds().SetId(1, int(b))
            wire_cells.InsertNextCell(line)
        self._wire_poly.SetPoints(wire_points)
        self._wire_poly.SetLines(wire_cells)
        self._set_cell_colors(self._wire_poly, wire_colors)
        self._wire_poly.Modified()
        self._update_arrow()

    def _make_actor(self, poly: vtk.vtkPolyData, wireframe: bool) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if wireframe:
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(self.line_width)
            prop.SetOpacity(1.0)
        else:
            prop.SetRepresentationToSurface()
            prop.SetOpacity(self.face_opacity)
        actor.SetPickable(True)
        return actor

    def _make_arrow_actor(self) -> vtk.vtkActor:
        source = vtk.vtkArrowSource()
        source.SetTipLength(0.28)
        source.SetTipRadius(0.055)
        source.SetShaftRadius(0.018)
        source.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(c / 255.0 for c in self.selection_color))
        actor.GetProperty().SetAmbient(0.35)
        actor.SetPickable(True)
        actor.SetVisibility(False)
        return actor

    def _gizmo_world_scale(self, origin: NDArray[np.float64]) -> float:
        if self.renderer is None:
            return 1.0
        renderer = self.renderer
        camera = renderer.GetActiveCamera()
        _width, height = renderer.GetRenderWindow().GetSize()
        viewport_h = max(1, int(height))
        if camera.GetParallelProjection():
            world_per_px = 2.0 * float(camera.GetParallelScale()) / float(viewport_h)
        else:
            fov_rad = math.radians(float(camera.GetViewAngle()))
            cam_pos = np.asarray(camera.GetPosition(), dtype=np.float64)
            distance = max(float(np.linalg.norm(origin - cam_pos)), 1e-6)
            world_per_px = 2.0 * distance * math.tan(fov_rad / 2.0) / float(viewport_h)
        return max(float(world_per_px * 72.0), 1e-6)

    @staticmethod
    def _matrix_for_arrow(origin: NDArray[np.float64], normal: NDArray[np.float64], scale: float) -> vtk.vtkMatrix4x4:
        x_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(x_axis, up))) > 0.95:
            up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        z_axis = np.cross(x_axis, up)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        matrix_np = np.eye(4, dtype=np.float64)
        matrix_np[:3, 0] = x_axis * scale
        matrix_np[:3, 1] = y_axis * scale
        matrix_np[:3, 2] = z_axis * scale
        matrix_np[:3, 3] = origin
        matrix = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                matrix.SetElement(r, c, float(matrix_np[r, c]))
        return matrix

    def _update_arrow(self) -> None:
        if self._arrow_actor is None:
            return
        self._arrow_box_face = None
        if self._selected_face is None or not self.visible:
            self._arrow_actor.SetVisibility(False)
            return
        box_id, face = self._selected_face
        box = self.data.get_box(box_id)
        if box is None:
            self._arrow_actor.SetVisibility(False)
            return
        origin = box.face_center(face)
        normal = box.face_normal(face)
        scale = self._gizmo_world_scale(origin)
        self._arrow_actor.SetUserMatrix(self._matrix_for_arrow(origin, normal, scale))
        self._arrow_actor.SetVisibility(True)
        self._arrow_box_face = (box_id, face)

    def _build_pipeline(self, renderer: vtk.vtkRenderer, out_actors: list[vtk.vtkProp]) -> None:
        self._face_poly = vtk.vtkPolyData()
        self._wire_poly = vtk.vtkPolyData()
        self._update_polys()
        self._face_actor = self._make_actor(self._face_poly, wireframe=False)
        self._wire_actor = self._make_actor(self._wire_poly, wireframe=True)
        self._arrow_actor = self._make_arrow_actor()
        out_actors.extend([self._face_actor, self._wire_actor, self._arrow_actor])
        self._sync_selection_timer()

    def _update_selection_pulse(self) -> None:
        if not self._active_selection and self._selected_face is None:
            self._sync_selection_timer()
            return
        self._selection_phase = (self._selection_phase + 0.35) % (2.0 * math.pi)
        self._update_polys()
        if self._renderer is not None:
            self._renderer.GetRenderWindow().Render()

    def update(self) -> None:
        self._update_polys()

    def data_index_from_picked_id(self, sub_id: int) -> int:
        if 0 <= sub_id < len(self._face_render_refs):
            box_id, _face = self._face_render_refs[sub_id]
            for i, box in enumerate(self.data.boxes):
                if box.id == box_id:
                    return i
        return -1

    def data_index_from_pick(self, sub_id: int, prop: vtk.vtkProp | None, association: str | None = None) -> int:
        if prop is self._face_actor:
            return self.data_index_from_picked_id(sub_id)
        if prop is self._wire_actor and 0 <= sub_id < len(self._wire_render_refs):
            box_id = self._wire_render_refs[sub_id]
            for i, box in enumerate(self.data.boxes):
                if box.id == box_id:
                    return i
        return -1

    def face_from_pick(self, sub_id: int, prop: vtk.vtkProp | None) -> tuple[str, str] | None:
        if prop is self._face_actor and 0 <= sub_id < len(self._face_render_refs):
            return self._face_render_refs[sub_id]
        return None

    def arrow_face_from_prop(self, prop: vtk.vtkProp | None) -> tuple[str, str] | None:
        if prop is not None and prop is self._arrow_actor:
            return self._arrow_box_face
        return None

    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float, float, float]:
        if 0 <= sub_id < len(self._face_render_refs):
            box_id, face = self._face_render_refs[sub_id]
            box = self.data.get_box(box_id)
            if box is not None:
                center = box.face_center(face)
                return (float(center[0]), float(center[1]), float(center[2]))
        return (0.0, 0.0, 0.0)

    def world_xyz_from_pick(self, sub_id: int, prop: vtk.vtkProp | None, association: str | None = None) -> tuple[float, float, float]:
        if prop is self._face_actor:
            return self.world_xyz_from_picked_id(sub_id)
        idx = self.data_index_from_pick(sub_id, prop, association)
        if idx >= 0:
            center = self.data.boxes[idx].face_center("top")
            return (float(center[0]), float(center[1]), float(center[2]))
        return (0.0, 0.0, 0.0)

    def handle_viewport_left_click(self, ctx: Any, event: Any, pick_result: Any) -> bool:
        if pick_result.layer is not self:
            self.active_selection = None
            ctx.controller.layer_internal_sel_changed.emit(self)
            return True
        face = self.face_from_pick(int(pick_result.raw_element_idx or 0), pick_result.prop)
        if face is not None:
            self.selected_face = face
        elif isinstance(pick_result.element_idx, int) and 0 <= pick_result.element_idx < len(self.data.boxes):
            self.selected_face = None
            self.active_selection = {self.data.boxes[pick_result.element_idx].id}
        ctx.controller.layer_internal_sel_changed.emit(self)
        return True

    def set_visible(self, visible: bool) -> None:
        super().set_visible(visible)
        self._update_arrow()

    def on_detached(self) -> None:
        if self._selection_timer.isActive():
            self._selection_timer.stop()
