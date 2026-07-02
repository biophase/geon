from __future__ import annotations

import copy
import math
import weakref
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import vtk
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget
from vtk.util import numpy_support as ns  # type: ignore

from geon.config import theme
from geon.data.boundingbox import FACE_NAMES, BoundingBox
from geon.rendering.boundingbox import WIRE_EDGES, BoundingBoxLayer
from geon.rendering.pointcloud import PointCloudLayer
from geon.tools.base import Event, ModeTool
from geon.tools.command_manager import Command, LambdaCommand
from geon.util.resources import resource_path


@dataclass
class AddBoundingBoxCmd(Command):
    layer_ref: weakref.ReferenceType[BoundingBoxLayer]
    box: BoundingBox

    def execute(self) -> None:
        layer = self.layer_ref()
        if layer is None:
            return
        if layer.data.get_box(self.box.id) is None:
            layer.data.append_box(copy.deepcopy(self.box))
        layer.update()

    def undo(self) -> None:
        layer = self.layer_ref()
        if layer is None:
            return
        layer.data.remove_box(self.box.id)
        layer.update()


def _display_ray(renderer: vtk.vtkRenderer, pos: tuple[int, int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, y = pos

    def display_to_world(z: float) -> NDArray[np.float64]:
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        wx, wy, wz, ww = renderer.GetWorldPoint()
        if abs(float(ww)) < 1e-12:
            return np.asarray((wx, wy, wz), dtype=np.float64)
        return np.asarray((wx / ww, wy / ww, wz / ww), dtype=np.float64)

    p0 = display_to_world(0.0)
    p1 = display_to_world(1.0)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return p0, np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    return p0, direction / norm


def _point_on_z_plane(renderer: vtk.vtkRenderer, pos: tuple[int, int], z: float) -> NDArray[np.float64] | None:
    origin, direction = _display_ray(renderer, pos)
    denom = float(direction[2])
    if abs(denom) <= 1e-12:
        return None
    t = (float(z) - float(origin[2])) / denom
    return origin + direction * t


def _point_on_axis(
    renderer: vtk.vtkRenderer,
    pos: tuple[int, int],
    axis_origin: NDArray[np.float64],
    axis: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    ray_origin, ray_dir = _display_ray(renderer, pos)
    matrix = np.column_stack((axis, -ray_dir))
    rhs = ray_origin - axis_origin
    try:
        params, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return axis_origin + axis * float(params[0])


def _make_poly_actor(
    poly: vtk.vtkPolyData,
    *,
    color: tuple[int, int, int],
    opacity: float,
    line_width: float = 1.0,
    wireframe: bool = False,
) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*(c / 255.0 for c in color))
    prop.SetOpacity(float(opacity))
    prop.SetLineWidth(float(line_width))
    if wireframe:
        prop.SetRepresentationToWireframe()
    actor.SetPickable(True)
    return actor


def _make_box_polys(box: BoundingBox) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
    corners = box.corners().astype(np.float32)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(corners, deep=True))
    face_cells = vtk.vtkCellArray()
    quads = box.face_quads()
    for face in FACE_NAMES:
        quad = vtk.vtkQuad()
        for i, pid in enumerate(quads[face]):
            quad.GetPointIds().SetId(i, int(pid))
        face_cells.InsertNextCell(quad)
    face_poly = vtk.vtkPolyData()
    face_poly.SetPoints(vtk_points)
    face_poly.SetPolys(face_cells)

    wire_points = vtk.vtkPoints()
    wire_points.SetData(ns.numpy_to_vtk(corners, deep=True))
    wire_cells = vtk.vtkCellArray()
    for a, b in WIRE_EDGES:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(a))
        line.GetPointIds().SetId(1, int(b))
        wire_cells.InsertNextCell(line)
    wire_poly = vtk.vtkPolyData()
    wire_poly.SetPoints(wire_points)
    wire_poly.SetLines(wire_cells)
    return face_poly, wire_poly


@dataclass
class CreateHorizontalBoundingBoxTool(ModeTool):
    label: ClassVar = "create_horizontal_bounding_box"
    tooltip: ClassVar = "Create horizontal box"
    icon_path: ClassVar = resource_path("annotate.png")
    shortcut: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar[bool] = True
    show_in_toolbar: ClassVar[bool] = False
    cursor_icon_path: ClassVar = None
    keep_focus: ClassVar[bool] = False

    stage: int = field(default=1, init=False)
    c1: NDArray[np.float64] | None = field(default=None, init=False)
    c2: NDArray[np.float64] | None = field(default=None, init=False)
    c3: NDArray[np.float64] | None = field(default=None, init=False)
    bottom_z: float = field(default=0.0, init=False)
    top_z: float = field(default=1.0, init=False)
    draft_box: BoundingBox | None = field(default=None, init=False)
    cull_outside: bool = field(default=False, init=False)

    _preview_face_actor: vtk.vtkActor | None = field(default=None, init=False)
    _preview_wire_actor: vtk.vtkActor | None = field(default=None, init=False)
    _arrow_actor: vtk.vtkActor | None = field(default=None, init=False)
    _selected_face: str | None = field(default=None, init=False)
    _press_pos: tuple[int, int] | None = field(default=None, init=False)
    _stage_z_start_y: int | None = field(default=None, init=False)
    _stage_z_base: float = field(default=0.0, init=False)
    _drag_face: str | None = field(default=None, init=False)
    _drag_start_point: NDArray[np.float64] | None = field(default=None, init=False)
    _drag_box_old: BoundingBox | None = field(default=None, init=False)
    _accept_btn: QPushButton | None = field(default=None, init=False)
    _stage_label: QLabel | None = field(default=None, init=False)

    def _layer(self) -> BoundingBoxLayer | None:
        layer = self.ctx.scene.active_layer
        return layer if isinstance(layer, BoundingBoxLayer) else None

    def _selection_color(self) -> tuple[int, int, int]:
        window = self.ctx.viewer.window()
        prefs = getattr(window, "preferences", None)
        if prefs is not None:
            color = getattr(prefs, "selection_color", None)
            if isinstance(color, list) and len(color) == 3:
                return tuple(int(c) for c in color)  # type: ignore[return-value]
        return theme.SELECTION_COLOR

    def _viewport_text_color(self) -> tuple[float, float, float]:
        window = self.ctx.viewer.window()
        prefs = getattr(window, "preferences", None)
        color = getattr(prefs, "viewport_text_color", None) if prefs is not None else None
        if isinstance(color, list) and len(color) == 3:
            return tuple(float(c) / 255.0 if float(c) > 1.0 else float(c) for c in color)  # type: ignore[return-value]
        return theme.VIEWPORT_TEXT_COLOR

    def _active_scene_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        bounds: list[float] | None = None
        for layer in self.ctx.scene.layers.values():
            if not layer.visible:
                continue
            if layer is self._layer():
                continue
            b = layer.data.get_extents()
            if b is None or len(b) != 6 or b[0] > b[1]:
                continue
            if bounds is None:
                bounds = [float(v) for v in b]
            else:
                bounds[0] = min(bounds[0], float(b[0]))
                bounds[1] = max(bounds[1], float(b[1]))
                bounds[2] = min(bounds[2], float(b[2]))
                bounds[3] = max(bounds[3], float(b[3]))
                bounds[4] = min(bounds[4], float(b[4]))
                bounds[5] = max(bounds[5], float(b[5]))
        return tuple(bounds) if bounds is not None else None  # type: ignore[return-value]

    def _init_vertical_extents(self) -> None:
        bounds = self._active_scene_bounds()
        if bounds is None:
            self.bottom_z = -1.0
            self.top_z = 1.0
            return
        self.bottom_z = float(bounds[4])
        self.top_z = float(bounds[5])
        if self.top_z <= self.bottom_z:
            self.top_z = self.bottom_z + 1.0

    def _set_top_view(self) -> None:
        renderer = self.ctx.viewer._renderer
        camera = renderer.GetActiveCamera()
        bounds = self._active_scene_bounds()
        if bounds is None:
            bounds = (-5.0, 5.0, -5.0, 5.0, -1.0, 1.0)
        center = (
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        )
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
        camera.SetParallelProjection(True)
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0], center[1], center[2] + span)
        camera.SetViewUp(0.0, 1.0, 0.0)
        renderer.ResetCamera(bounds)
        renderer.ResetCameraClippingRange()
        self.ctx.viewer.clear_viewport_text()
        self.ctx.viewer.rerender()

    def _is_top_view(self) -> bool:
        camera = self.ctx.viewer._renderer.GetActiveCamera()
        dop = np.asarray(camera.GetDirectionOfProjection(), dtype=np.float64)
        up = np.asarray(camera.GetViewUp(), dtype=np.float64)
        return (
            bool(camera.GetParallelProjection())
            and float(np.dot(dop, np.asarray((0.0, 0.0, -1.0)))) > 0.995
            and float(np.dot(up, np.asarray((0.0, 1.0, 0.0)))) > 0.995
        )

    def _update_top_warning(self) -> None:
        if self.stage <= 3 and not self._is_top_view():
            self.ctx.viewer.show_viewport_text(
                "Currently drawing on XY plane but view is not set to 'top'",
                color=self._viewport_text_color(),
            )
        else:
            self.ctx.viewer.clear_viewport_text()

    def _world_per_pixel(self, origin: NDArray[np.float64]) -> float:
        renderer = self.ctx.viewer._renderer
        camera = renderer.GetActiveCamera()
        _width, height = renderer.GetRenderWindow().GetSize()
        h = max(1, int(height))
        if camera.GetParallelProjection():
            return 2.0 * float(camera.GetParallelScale()) / float(h)
        distance = max(float(np.linalg.norm(origin - np.asarray(camera.GetPosition()))), 1e-6)
        return 2.0 * distance * math.tan(math.radians(camera.GetViewAngle()) * 0.5) / h

    def _xy_point(self, event: Event) -> NDArray[np.float64] | None:
        return _point_on_z_plane(self.ctx.viewer._renderer, event.pos, 0.0)

    def _update_xy_preview(self, event: Event) -> None:
        p = self._xy_point(event)
        if p is None:
            return
        if self.stage == 2 and self.c1 is not None:
            self.draft_box = self._make_stage2_preview_box(p)
            self._update_box_preview()
        elif self.stage == 3 and self.c1 is not None and self.c2 is not None:
            self.draft_box = self._make_stage3_preview_box(p)
            self._update_box_preview()

    def _make_stage2_preview_box(self, point: NDArray[np.float64]) -> BoundingBox | None:
        if self.c1 is None:
            return None
        p1 = np.asarray((self.c1[0], self.c1[1]), dtype=np.float64)
        p2 = np.asarray((point[0], point[1]), dtype=np.float64)
        v = p2 - p1
        width = float(np.linalg.norm(v))
        if width <= 1e-12:
            return None
        yaw = math.atan2(float(v[1]), float(v[0]))
        center = p1 + 0.5 * v
        bounds = self._active_scene_bounds()
        span_xy = 1.0 if bounds is None else max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1.0)
        depth = max(span_xy * 0.0025, 1e-4)
        return BoundingBox(
            center_bottom_xyz=(float(center[0]), float(center[1]), float(self.bottom_z)),
            yaw=yaw,
            width=width,
            depth=depth,
            height=max(float(self.top_z - self.bottom_z), 1e-6),
        )

    def _make_stage3_preview_box(self, point: NDArray[np.float64]) -> BoundingBox | None:
        if self.c1 is None or self.c2 is None:
            return None
        try:
            return BoundingBox.from_horizontal_corners(
                tuple(float(v) for v in self.c1),
                tuple(float(v) for v in self.c2),
                tuple(float(v) for v in point),
                self.bottom_z,
                self.top_z,
            )
        except ValueError:
            return None

    def _make_draft_from_state(self) -> BoundingBox | None:
        if self.c1 is None or self.c2 is None or self.c3 is None:
            return None
        try:
            return BoundingBox.from_horizontal_corners(
                tuple(float(v) for v in self.c1),
                tuple(float(v) for v in self.c2),
                tuple(float(v) for v in self.c3),
                self.bottom_z,
                self.top_z,
            )
        except ValueError:
            return None

    def _ensure_preview_actors(self) -> None:
        if self._preview_face_actor is None:
            self._preview_face_actor = _make_poly_actor(
                vtk.vtkPolyData(),
                color=self._selection_color(),
                opacity=0.25,
                wireframe=False,
            )
            self.ctx.viewer._renderer.AddActor(self._preview_face_actor)
        if self._preview_wire_actor is None:
            self._preview_wire_actor = _make_poly_actor(
                vtk.vtkPolyData(),
                color=self._selection_color(),
                opacity=1.0,
                line_width=2.5,
                wireframe=True,
            )
            self.ctx.viewer._renderer.AddActor(self._preview_wire_actor)

    def _update_box_preview(self) -> None:
        if self.draft_box is None:
            self.draft_box = self._make_draft_from_state()
        if self.draft_box is None:
            self._hide_box_preview()
            return
        self._ensure_preview_actors()
        face_poly, wire_poly = _make_box_polys(self.draft_box)
        assert self._preview_face_actor is not None
        assert self._preview_wire_actor is not None
        self._preview_face_actor.GetMapper().SetInputData(face_poly)
        self._preview_wire_actor.GetMapper().SetInputData(wire_poly)
        self._preview_face_actor.GetMapper().Update()
        self._preview_wire_actor.GetMapper().Update()
        self._preview_face_actor.SetVisibility(True)
        self._preview_wire_actor.SetVisibility(True)
        self._update_arrow()
        self._update_clipping()
        self.ctx.viewer.rerender()

    def _hide_box_preview(self) -> None:
        for actor in (self._preview_face_actor, self._preview_wire_actor, self._arrow_actor):
            if actor is not None:
                actor.SetVisibility(False)
        self._clear_clipping()
        self.ctx.viewer.rerender()

    def _set_draft_box(self, box: BoundingBox | None) -> None:
        self.draft_box = copy.deepcopy(box) if box is not None else None
        if self.draft_box is None:
            self._hide_box_preview()
        else:
            self._update_box_preview()

    def _vertical_clipping_planes(self, box: BoundingBox) -> list[vtk.vtkPlane]:
        axis_x, axis_y, _axis_z = box.axes()
        center = np.asarray(box.center_bottom_xyz, dtype=np.float64) + axis_x * 0.0 + axis_y * 0.0
        specs = (
            (center - axis_x * (box.width * 0.5), axis_x),
            (center + axis_x * (box.width * 0.5), -axis_x),
            (center - axis_y * (box.depth * 0.5), axis_y),
            (center + axis_y * (box.depth * 0.5), -axis_y),
        )
        planes: list[vtk.vtkPlane] = []
        for origin, normal in specs:
            plane = vtk.vtkPlane()
            plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
            plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
            planes.append(plane)
        return planes

    def _update_clipping(self) -> None:
        if not self.cull_outside or self.draft_box is None or self.stage < 4:
            self._clear_clipping()
            return
        planes = self._vertical_clipping_planes(self.draft_box)
        for layer in self.ctx.scene.layers.values():
            if isinstance(layer, PointCloudLayer):
                layer.set_clipping_planes(planes)

    def _clear_clipping(self) -> None:
        for layer in self.ctx.scene.layers.values():
            if isinstance(layer, PointCloudLayer):
                layer.clear_clipping_planes()

    def _set_stage(self, stage: int) -> None:
        self.stage = int(stage)
        self.ctx.viewer.set_camera_enabled(True)
        if self._stage_label is not None:
            self._stage_label.setText(f"Stage: {self.stage}")
        if self._accept_btn is not None:
            self._accept_btn.setEnabled(self.draft_box is not None and self.stage >= 6)
        self._update_top_warning()

    def _record_local(self, title: str, execute, undo) -> None:
        self.command_manager.do(LambdaCommand(title, execute, undo))

    def activate(self) -> None:
        super().activate()
        if self._layer() is None:
            self.ctx.controller.deactivate_tool()
            return
        self._init_vertical_extents()
        self._set_top_view()
        self._set_stage(1)
        self.ctx.viewer.rerender()

    def deactivate(self) -> None:
        self._clear_clipping()
        self.ctx.viewer.clear_viewport_text()
        for actor in (self._preview_face_actor, self._preview_wire_actor, self._arrow_actor):
            if actor is not None:
                self.ctx.viewer._renderer.RemoveActor(actor)
        self._preview_face_actor = None
        self._preview_wire_actor = None
        self._arrow_actor = None
        self.ctx.viewer.set_camera_enabled(True)
        self.ctx.viewer.rerender()
        return super().deactivate()

    def left_button_press_hook(self, event: Event) -> None:
        self._press_pos = event.pos
        if self.stage >= 6:
            arrow_face = self._pick_arrow(event)
            if arrow_face is not None and self.draft_box is not None:
                self._drag_face = arrow_face
                normal = self.draft_box.face_normal(arrow_face)
                origin = self.draft_box.face_center(arrow_face)
                start = _point_on_axis(self.ctx.viewer._renderer, event.pos, origin, normal)
                if start is not None:
                    self._drag_start_point = start
                    self._drag_box_old = copy.deepcopy(self.draft_box)
                    self.ctx.viewer.set_camera_enabled(False)

    def left_button_release_hook(self, event: Event) -> None:
        if self._drag_face is not None:
            self._finish_face_drag(event)
            self._press_pos = None
            return
        is_click = self._press_pos is None or (
            (event.pos[0] - self._press_pos[0]) ** 2 + (event.pos[1] - self._press_pos[1]) ** 2 <= 16
        )
        if not is_click:
            self._press_pos = None
            return
        if self.stage <= 3:
            p = self._xy_point(event)
            if p is None:
                self._press_pos = None
                return
            self._commit_xy_point(p)
        elif self.stage == 4:
            old = self.bottom_z
            new = float(self.bottom_z)

            def execute() -> None:
                self._set_bottom_z(new)
                self._set_stage(5)
                self._stage_z_start_y = None
                self._stage_z_base = self.top_z

            def undo() -> None:
                self._set_bottom_z(old)
                self._set_stage(4)
                self._stage_z_start_y = None
                self._stage_z_base = old

            self._record_local(
                "Set bottom bounding-box plane",
                execute,
                undo,
            )
        elif self.stage == 5:
            old = self.top_z
            new = float(self.top_z)

            def execute() -> None:
                self._set_top_z(new)
                self.draft_box = self._make_draft_from_state()
                self._set_stage(6)
                self._update_box_preview()

            def undo() -> None:
                self._set_top_z(old)
                self._set_stage(5)
                self._stage_z_start_y = None
                self._stage_z_base = old

            self._record_local(
                "Set top bounding-box plane",
                execute,
                undo,
            )
        elif self.stage >= 6:
            face = self._pick_preview_face(event)
            self._selected_face = face
            self._update_arrow()
            self.ctx.viewer.rerender()
        self._press_pos = None

    def _commit_xy_point(self, point: NDArray[np.float64]) -> None:
        if self.stage == 1:
            old = self.c1
            new = point.copy()

            def execute() -> None:
                self.c1 = new
                self.c2 = None
                self.c3 = None
                self._set_draft_box(None)
                self._set_stage(2)

            def undo() -> None:
                self.c1 = old
                self.c2 = None
                self.c3 = None
                self._set_draft_box(None)
                self._set_stage(1)

            self._record_local("Set first bounding-box corner", execute, undo)
        elif self.stage == 2:
            old = self.c2
            new = point.copy()
            preview_box = self._make_stage2_preview_box(new)

            def execute() -> None:
                self.c2 = new
                self.c3 = None
                self._set_draft_box(preview_box)
                self._set_stage(3)

            def undo() -> None:
                self.c2 = old
                self.c3 = None
                self._set_draft_box(None)
                self._set_stage(2)

            self._record_local("Set second bounding-box corner", execute, undo)
        elif self.stage == 3:
            old = self.c3
            new = point.copy()
            preview_box = self._make_stage3_preview_box(new)

            def execute() -> None:
                self.c3 = new
                self._set_draft_box(preview_box)
                self._stage_z_start_y = None
                self._stage_z_base = 0.0
                self._set_stage(4)

            def undo() -> None:
                self.c3 = old
                self._set_draft_box(None)
                self._set_stage(3)

            self._record_local("Set third bounding-box corner", execute, undo)

    def _set_bottom_z(self, z: float) -> None:
        self.bottom_z = float(z)
        self.draft_box = self._make_draft_from_state()
        self._update_box_preview()

    def _set_top_z(self, z: float) -> None:
        self.top_z = float(z)
        self.draft_box = self._make_draft_from_state()
        self._update_box_preview()

    def mouse_move_event_hook(self, event: Event) -> None:
        self._update_top_warning()
        if self.stage in (2, 3):
            self._update_xy_preview(event)
        elif self.stage in (4, 5):
            if self._press_pos is not None:
                return
            if self.draft_box is None:
                self.draft_box = self._make_draft_from_state()
            if self.draft_box is None:
                return
            center = self.draft_box.face_center("bottom")
            if self._stage_z_start_y is None:
                self._stage_z_start_y = event.pos[1]
                self._stage_z_base = self.bottom_z if self.stage == 4 else self.top_z
            dz = (event.pos[1] - self._stage_z_start_y) * self._world_per_pixel(center)
            if self.stage == 4:
                self.bottom_z = self._stage_z_base + dz
            else:
                self.top_z = self._stage_z_base + dz
            self.draft_box = self._make_draft_from_state()
            self._update_box_preview()
        elif self._drag_face is not None:
            self._update_face_drag(event)

    def _pick_preview_face(self, event: Event) -> str | None:
        if self._preview_face_actor is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0015)
        picker.Pick(event.pos[0], event.pos[1], 0, self.ctx.viewer._renderer)
        if picker.GetViewProp() is not self._preview_face_actor:
            return None
        cell_id = int(picker.GetCellId())
        if 0 <= cell_id < len(FACE_NAMES):
            return FACE_NAMES[cell_id]
        return None

    def _pick_arrow(self, event: Event) -> str | None:
        if self._arrow_actor is None or self._selected_face is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(event.pos[0], event.pos[1], 0, self.ctx.viewer._renderer)
        if picker.GetViewProp() is self._arrow_actor:
            return self._selected_face
        return None

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
        actor.GetProperty().SetColor(*(c / 255.0 for c in self._selection_color()))
        actor.GetProperty().SetAmbient(0.35)
        actor.SetPickable(True)
        return actor

    def _update_arrow(self) -> None:
        if self.draft_box is None or self._selected_face is None:
            if self._arrow_actor is not None:
                self._arrow_actor.SetVisibility(False)
            return
        if self._arrow_actor is None:
            self._arrow_actor = self._make_arrow_actor()
            self.ctx.viewer._renderer.AddActor(self._arrow_actor)
        origin = self.draft_box.face_center(self._selected_face)
        normal = self.draft_box.face_normal(self._selected_face)
        matrix = BoundingBoxLayer._matrix_for_arrow(
            origin,
            normal,
            max(self._world_per_pixel(origin) * 72.0, 1e-6),
        )
        self._arrow_actor.SetUserMatrix(matrix)
        self._arrow_actor.SetVisibility(True)

    def _update_face_drag(self, event: Event) -> None:
        if self.draft_box is None or self._drag_face is None or self._drag_start_point is None or self._drag_box_old is None:
            return
        normal = self._drag_box_old.face_normal(self._drag_face)
        origin = self._drag_box_old.face_center(self._drag_face)
        current = _point_on_axis(self.ctx.viewer._renderer, event.pos, origin, normal)
        if current is None:
            return
        delta = float(np.dot(current - self._drag_start_point, normal))
        self.draft_box = copy.deepcopy(self._drag_box_old)
        self.draft_box.adjust_face(self._drag_face, delta)
        self._update_box_preview()

    def _finish_face_drag(self, event: Event) -> None:
        old_box = copy.deepcopy(self._drag_box_old)
        new_box = copy.deepcopy(self.draft_box)
        face = self._drag_face
        self._drag_face = None
        self._drag_start_point = None
        self._drag_box_old = None
        self.ctx.viewer.set_camera_enabled(True)
        if old_box is None or new_box is None or face is None:
            return

        def set_box(box: BoundingBox) -> None:
            self.draft_box = copy.deepcopy(box)
            self._update_box_preview()

        self._record_local(
            "Adjust bounding-box face",
            lambda: set_box(new_box),
            lambda: set_box(old_box),
        )

    def _accept(self) -> None:
        layer = self._layer()
        if layer is None or self.draft_box is None:
            return
        cmd = AddBoundingBoxCmd("Add bounding box", weakref.ref(layer), copy.deepcopy(self.draft_box))
        self.ctx.controller.do_global(cmd)
        layer.active_selection = {self.draft_box.id}
        window = self.ctx.viewer.window()
        mark_modified = getattr(window, "_mark_active_doc_modified", None)
        if callable(mark_modified):
            mark_modified()
        self.ctx.controller.scene_tree_request_change.emit()
        self.ctx.controller.layer_internal_sel_changed.emit(layer)
        self.ctx.controller.deactivate_tool()

    def _cancel(self) -> None:
        self.ctx.controller.deactivate_tool()

    def _toggle_cull(self, checked: bool) -> None:
        self.cull_outside = bool(checked)
        self._update_clipping()
        self.ctx.viewer.rerender()

    def create_context_widget(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        layout = QHBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(2, 1, 2, 1)
        self._stage_label = QLabel(f"Stage: {self.stage}", w)
        layout.addWidget(self._stage_label)
        top_btn = QPushButton("Top view", w)
        top_btn.setIcon(QIcon(resource_path("camera_top.png")))
        top_btn.pressed.connect(self._set_top_view)
        layout.addWidget(top_btn)
        cull = QCheckBox("Cull outside", w)
        cull.setChecked(self.cull_outside)
        cull.toggled.connect(self._toggle_cull)
        layout.addWidget(cull)
        self._accept_btn = QPushButton("Accept", w)
        self._accept_btn.setEnabled(False)
        self._accept_btn.pressed.connect(self._accept)
        layout.addWidget(self._accept_btn)
        cancel = QPushButton("Cancel", w)
        cancel.pressed.connect(self._cancel)
        layout.addWidget(cancel)
        return w
