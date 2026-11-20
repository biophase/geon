import numpy as np
import vtk
from vtk.util import numpy_support as ns

from geon.config.theme import DEFAULT_OBJ_COLOR
from geon.data.mesh import MeshData, MeshLevel
from geon.data.pointcloud import SemanticSegmentation, InstanceSegmentation
from .selection import SelectableLayer
from .base import BaseLayer
from .layer_registry import layer_for


@layer_for(MeshData)
class MeshLayer(BaseLayer[MeshData], SelectableLayer):
    @classmethod
    def merge(cls, layers, *, name=None):
        merged,first=cls._merge_layers(layers,name)
        merged.set_active_field(first.active_level,first.active_field_name)
        merged.opacity=first.opacity
        return merged

    layer_type_id = "mesh"
    @property
    def use_cell_picking_for_selection(self):
        return self.active_level != MeshLevel.VERTEX

    def __init__(self, data, browser_name=None):
        super().__init__(data)
        if browser_name is not None:
            self.data.name = str(browser_name)
        self._poly = None
        self._mapper = None
        self._actor = None
        self._active_level = MeshLevel.FACE
        self._active_field_name = None
        self._active_selection = None
        self.opacity = 1.0
        self.is_preview = False
        self._color_overrides = {}
        self._color_mapping = None
        self._visibility_mask = None
        self.unassigned_point_color = (204, 204, 204, 255)
        self.preview_field = None
        self.preview_map_changed = None

    def _build_pipeline(self, renderer, out_actors):
        self._poly = vtk.vtkPolyData()
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputData(self._poly)
        self._mapper.ScalarVisibilityOff()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().SetColor(*DEFAULT_OBJ_COLOR[:3])
        self._actor.SetVisibility(self.visible)
        self._element_poly = vtk.vtkPolyData()
        self._element_mapper = vtk.vtkPolyDataMapper()
        self._element_mapper.SetInputData(self._element_poly)
        self._element_actor = vtk.vtkActor()
        self._element_actor.SetMapper(self._element_mapper)
        self._element_actor.GetProperty().SetLineWidth(3)
        self._element_actor.GetProperty().SetPointSize(7)
        self._element_actor.GetProperty().SetColor(.3, .6, .7)
        self._selection_poly = vtk.vtkPolyData()
        selection_mapper = vtk.vtkPolyDataMapper()
        selection_mapper.SetInputData(self._selection_poly)
        selection_mapper.ScalarVisibilityOff()
        self._selection_actor = vtk.vtkActor()
        self._selection_actor.SetMapper(selection_mapper)
        self._selection_actor.SetPickable(False)
        self._selection_actor.GetProperty().SetColor(1, .5, 0)
        self._selection_actor.GetProperty().SetOpacity(.65)
        self._selection_actor.GetProperty().SetPointSize(10)
        self._selection_actor.GetProperty().SetLineWidth(5)
        self.update()
        out_actors.extend([self._actor, self._element_actor, self._selection_actor])

    def update(self):
        if self._poly is None:
            return
        points = vtk.vtkPoints()
        points.SetData(ns.numpy_to_vtk(self.data.vertices, deep=True))
        offsets = np.concatenate(([0], np.cumsum([len(f) for f in self.data.faces], dtype=np.int64)))
        indices = np.concatenate(self.data.faces) if self.data.faces else np.empty(0, dtype=np.int64)
        cells = vtk.vtkCellArray()
        cells.SetData(ns.numpy_to_vtkIdTypeArray(offsets, deep=True),
                      ns.numpy_to_vtkIdTypeArray(indices, deep=True))
        self._poly.SetPoints(points)
        self._poly.SetPolys(cells)
        self._poly.Modified()
        self._update_display()

    def data_index_from_picked_id(self, sub_id):
        if not 0 <= sub_id < len(self.data.vertices):
            raise IndexError(sub_id)
        return int(sub_id)

    def world_xyz_from_picked_id(self, sub_id):
        return tuple(float(v) for v in self.data.vertices[self.data_index_from_picked_id(sub_id)])

    def data_index_from_pick(self, sub_id, prop, association):
        if prop is self._element_actor:
            return int(sub_id)
        if association == "cell":
            if self.active_level == MeshLevel.VERTEX:
                return int(self.data.faces[sub_id][0])
            if not 0 <= sub_id < len(self.data.faces):
                raise IndexError(sub_id)
            return int(sub_id)
        return self.data_index_from_picked_id(sub_id)

    def world_xyz_from_pick(self, sub_id, prop, association):
        index = self.data_index_from_pick(sub_id, prop, association)
        if prop is self._element_actor:
            xyz = (self.data.vertices[index] if self.active_level == MeshLevel.VERTEX else
                   self.data.vertices[self.data.edges[index]].mean(axis=0))
            return tuple(float(v) for v in xyz)
        if association == "cell" and self.active_level != MeshLevel.VERTEX:
            return tuple(float(v) for v in self.data.vertices[self.data.faces[index]].mean(axis=0))
        return self.world_xyz_from_picked_id(index)

    @property
    def active_level(self):
        return self._active_level

    @property
    def active_field(self):
        if self.preview_field is not None:
            return self.preview_field
        if self._active_field_name is None:
            return None
        fields = self.data.get_fields(names=self._active_field_name, level=self.active_level)
        return fields[0] if fields else None

    @property
    def active_field_name(self):
        return self._active_field_name

    def set_active_field(self, level, name=None):
        level = MeshLevel(level)
        if name is not None and not self.data.get_fields(names=name, level=level):
            raise ValueError(f"Unknown {level.value} field: {name}")
        if self._active_level != level:
            self._active_selection = None
        self._active_level = level
        self._active_field_name = name
        self.preview_field = None
        self.update()
        self.signals.activeFieldChanged.emit()

    @property
    def active_selection(self):
        return self._active_selection

    @active_selection.setter
    def active_selection(self, value):
        if value is not None:
            value = np.asarray(value)
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError("Mesh selection must contain integer element indices.")
            if np.any(value < 0) or np.any(value >= self.data.element_count(self.active_level)):
                raise ValueError("Mesh selection index is out of range.")
            value = np.unique(value).astype(np.int64)
        self._active_selection = value
        self.update()
        self.signals.selectionChanged.emit()

    def capture_selection(self):
        return self.active_level, None if self.active_selection is None else self.active_selection.copy()

    def restore_selection(self, state):
        level, indices = state
        if self.active_level != level:
            self.set_active_field(level)
        self.active_selection = indices

    def set_opacity(self, opacity):
        self.opacity = max(0., min(1., float(opacity)))
        if self._actor is not None:
            self._actor.GetProperty().SetOpacity(self.opacity)

    @staticmethod
    def _colors(field):
        if isinstance(field, SemanticSegmentation):
            return (field.schema.get_color_array(field.data) if len(field.data)
                    else np.empty((0, 3), dtype=np.uint8))
        if isinstance(field, InstanceSegmentation):
            colors = field.get_color_array()
            colors[field.data.reshape(-1) == -1] = (204, 204, 204)
            return colors
        from geon.rendering.color_mapping import default_map, components
        from geon.data.definitions import VectorColorMap, finite_extrema, resolve_limits, sample_map
        mapping = default_map(field)
        values = components(field, mapping)
        if isinstance(mapping, VectorColorMap):
            channels = []
            for values, curve in zip(values, mapping.channels):
                low, high = resolve_limits(curve, finite_extrema(values))
                positions = np.full(len(values), .5) if high == low else (values-low)/(high-low)
                channels.append(np.interp(np.clip(positions, 0, 1), curve.color_positions, curve.output_values))
            return np.clip(np.column_stack(channels)*255, 0, 255).astype(np.uint8)
        low, high = resolve_limits(mapping, finite_extrema(values[0]))
        positions = np.full(len(values[0]), .5) if high == low else (values[0]-low)/(high-low)
        return np.clip(sample_map(mapping, np.nan_to_num(positions))*255, 0, 255).astype(np.uint8)

    def _update_display(self):
        self._mapper.ScalarVisibilityOff()
        self._element_mapper.ScalarVisibilityOff()
        self._element_actor.SetVisibility(False)
        field = self.active_field
        if field is not None:
            field = self._color_overrides.get(field.name, field)
        level = self.active_level
        self._actor.SetPickable(level != MeshLevel.EDGE)
        if level in (MeshLevel.VERTEX, MeshLevel.EDGE):
            self._element_poly.SetPoints(self._poly.GetPoints())
            if level == MeshLevel.VERTEX:
                cells = vtk.vtkCellArray()
                for i in range(len(self.data.vertices)):
                    cells.InsertNextCell(1, [i])
                self._element_poly.SetLines(None)
                self._element_poly.SetVerts(cells)
            else:
                cells = vtk.vtkCellArray()
                for edge in self.data.edges:
                    cells.InsertNextCell(2, edge)
                self._element_poly.SetVerts(None)
                self._element_poly.SetLines(cells)
            self._element_poly.Modified()
            self._element_actor.SetVisibility(self.visible)
        if self.preview_field is not None:
            if self.preview_map_changed is not None:
                self.preview_map_changed(field)
        elif field is not None:
            colors = self._colors(field)
            if level == MeshLevel.OBJECT:
                colors = colors[np.searchsorted(self.data.object_ids, self.data.face_object_ids)]
            array = ns.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            array.SetName("Colors")
            if level == MeshLevel.VERTEX:
                self._poly.GetPointData().SetScalars(array)
                self._mapper.SetScalarModeToUsePointData()
                self._mapper.ScalarVisibilityOn()
            elif level in (MeshLevel.FACE, MeshLevel.OBJECT):
                self._poly.GetCellData().SetScalars(array)
                self._mapper.SetScalarModeToUseCellData()
                self._mapper.ScalarVisibilityOn()
            else:
                self._element_poly.GetCellData().SetScalars(array)
                self._element_mapper.SetScalarModeToUseCellData()
                self._element_mapper.ScalarVisibilityOn()
        self._selection_poly.Initialize()
        selected = self.active_selection
        if selected is not None and len(selected):
            self._selection_poly.SetPoints(self._poly.GetPoints())
            cells = vtk.vtkCellArray()
            if level == MeshLevel.OBJECT:
                selected = np.flatnonzero(np.isin(self.data.face_object_ids, self.data.object_ids[selected]))
            for index in selected:
                indices = ([int(index)] if level == MeshLevel.VERTEX else
                           self.data.edges[index] if level == MeshLevel.EDGE else self.data.faces[index])
                cells.InsertNextCell(len(indices), indices)
            if level == MeshLevel.VERTEX:
                self._selection_poly.SetVerts(cells)
            elif level == MeshLevel.EDGE:
                self._selection_poly.SetLines(cells)
            else:
                self._selection_poly.SetPolys(cells)
        self._selection_actor.SetVisibility(self.visible and self.has_selection)
        self._selection_poly.Modified()
        self.set_opacity(self.opacity)

    def _apply_visibility(self):
        super()._apply_visibility()
        if self._actor is not None:
            self._element_actor.SetVisibility(self.visible and self.active_level in (MeshLevel.VERTEX, MeshLevel.EDGE))
            self._selection_actor.SetVisibility(self.visible and self.has_selection)

    def handle_viewport_left_click(self, ctx, event, pick_result):
        if self.is_preview or pick_result.layer is not self or pick_result.element_idx is None:
            return False
        import weakref
        from geon.tools.command_manager import LambdaCommand
        level = self.active_level
        index = int(pick_result.element_idx)
        if level == MeshLevel.OBJECT:
            index = int(np.searchsorted(self.data.object_ids, self.data.face_object_ids[index]))
        if not 0 <= index < self.data.element_count(level):
            return False
        previous = self.capture_selection()
        selected = set(self.active_selection if self.active_selection is not None else ()) if event.shift else set()
        if index in selected:
            selected.remove(index)
        else:
            selected.add(index)
        current = (level, np.array(sorted(selected), dtype=np.int64))
        layer_ref = weakref.ref(self)
        def apply(state):
            layer = layer_ref()
            if layer is not None:
                name = layer.active_field_name if layer.active_level == state[0] else None
                layer.restore_selection(state)
                layer.set_active_field(state[0], name)
                ctx.controller.layer_internal_sel_changed.emit(layer)
                ctx.viewer.rerender()
        ctx.controller.do_global(LambdaCommand("Select mesh elements", lambda: apply(current), lambda: apply(previous)))
        return True

    def preview_color_map(self, field, mapping):
        self._color_overrides[field.name] = field
        self.update()

    def extract_selection(self):
        if not self.has_selection:
            raise ValueError("The mesh has no selected elements.")
        data = self.data.extract_selection(self.active_selection, level=self.active_level)
        return MeshLayer(data)
