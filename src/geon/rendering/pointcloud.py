from geon.util.operation_timing import timed, timing  # TEMP_VISIBILITY_PROFILE
import vtk 
from vtk.util import numpy_support as ns # type: ignore


import numpy as np
from numpy.typing import NDArray

from geon.data.pointcloud import (PointCloudData, FieldType, 
                                  SemanticSegmentation, InstanceSegmentation,
                                  FieldBase
                                  )

from geon.config import theme
from geon.settings import DEFAULT_PREFS

from .base import BaseLayer
from .pointcloud_blend import PointCloudBlend
from .pointcloud_lod import PointCloudLOD
from .layer_registry import layer_for


from typing import Optional, Tuple
import math
from .pointcloud_visibility import PointCloudVisibility
from .pointcloud_selection import PointCloudSelection
from .color_mapping import GPUColorMapping, default_map, components
from geon.data.definitions import VectorColorMap, finite_extrema, resolve_limits, sample_map



@layer_for(PointCloudData)
class PointCloudLayer(BaseLayer[PointCloudData]):
    layer_type_id = "pointcloud"

    def __init__(self, data:PointCloudData, browser_name = 'Point Cloud'):
        super().__init__(data)
        self._poly:         Optional[vtk.vtkPolyData] = None
        self._poly_coarse:  Optional[vtk.vtkPolyData] = None

        self._active_field_name:    Optional[str] = None
        self.background_field_name: str | None = None
        self.foreground_mix: float = 0.5
        self._blend: PointCloudBlend | None = None
        self._lod: PointCloudLOD | None = None
        self._coarse_stride = 100
        self._visibility_shader = None
        self._color_mapping = None
        self._color_overrides = {}
        self._visibility_mask:      Optional[NDArray[np.bool_]] = None
        self._active_selection:     Optional[NDArray[np.int32]] = None
        self._selection_shader = PointCloudSelection(self)
        
        self._mapper_fine:      Optional[vtk.vtkMapper] = None
        self._mapper_coarse:    Optional[vtk.vtkMapper] = None
        self._clipping_planes: list[vtk.vtkPlane] = []

        self._main_actor: Optional[vtk.vtkActor] = None

        
        self.browser_name = browser_name
        self.point_size: int = 2
        self.unassigned_point_color = tuple(DEFAULT_PREFS["unassigned_point_color"])
        
        

        

    

    
    def set_unassigned_point_color(self, color: tuple[int, ...] | list[int]) -> None:
        rgba = tuple(int(max(0, min(255, c))) for c in color)
        if len(rgba) != 4:
            raise ValueError("Unassigned point color requires four RGBA channels")
        if rgba != self.unassigned_point_color:
            self.unassigned_point_color = rgba
            self.update()

    @property
    def active_field(self) -> FieldBase | None:
        name = self.active_field_name
        if name is None:
            return None
        f = self.data.get_fields(name)
        if len(f) == 0:
            return None
        return f[0]
        
    @property
    def vf_active_index (self) -> dict[str, int]:
        return {field.name: field.scalar_component for field in self.data.get_fields()}

    @property
    def active_field_name(self) -> Optional[str]:
        return self._active_field_name
    

    @active_field_name.setter
    def active_field_name(self, field_name:str) -> None:
        assert field_name in self.data.field_names
        self._active_field_name = field_name
        self.signals.activeFieldChanged.emit()


    @property
    def active_selection(self) -> Optional[NDArray[np.int32]]:
        return self._active_selection
    
    @active_selection.setter
    def active_selection(self, selection: Optional[NDArray[np.int32]]) -> None:
        if selection is not None and selection.size:
            assert selection.max() < self.data.points.shape[0]
            assert selection.min() >= 0
        self._active_selection = selection
        self._selection_shader.update_mask()
        
        
        
    
    
    def set_active_field_name(self, name:str):
        self.active_field_name = name
        self.update()

    def set_background_field_name(self, name: str | None) -> None:
        if name is not None and name not in self.data.field_names:
            raise KeyError(f"Field '{name}' not found.")
        self.background_field_name = name
        self.update()

    def set_foreground_mix(self, amount: float) -> None:
        amount = float(amount)
        if not math.isfinite(amount):
            raise ValueError("Blend amount must be finite.")
        self.foreground_mix = max(0.0, min(1.0, amount))
        if self._color_mapping is not None:
            self._color_mapping.set_mix(self.foreground_mix)

    def set_vector_field_active_index(self, field_name: str, index: int) -> int:
        fields = self.data.get_fields(names=field_name)
        if not fields:
            raise KeyError(f"Field '{field_name}' not found.")
        field = fields[0]
        if field.field_type not in {FieldType.VECTOR, FieldType.NORMAL}:
            raise ValueError(f"Field '{field_name}' is not a vector-like field.")
        if field.data.ndim < 2:
            raise ValueError(f"Field '{field_name}' has no component axis.")
        max_index = max(0, field.data.shape[1] - 1)
        clamped = int(max(0, min(max_index, index)))
        field.scalar_component = clamped
        self.update()
        return clamped

    @timed('Set visibility mask')  # TEMP_VISIBILITY_PROFILE
    def set_visibility_mask(self, mask: Optional[NDArray[np.bool_]]) -> None:
        if mask is None:
            self._visibility_mask = None
            if self._visibility_shader is not None:
                self._visibility_shader.set_mask(None)
            self.signals.pointVisibilityChanged.emit()
            return
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 1:
            raise ValueError(f"Expected 1D visibility mask, got shape {mask_arr.shape}")
        if mask_arr.shape[0] != self.data.points.shape[0]:
            raise ValueError(
                "Visibility mask length does not match point count: "
                f"{mask_arr.shape[0]} vs {self.data.points.shape[0]}"
            )
        self._visibility_mask = mask_arr
        if self._visibility_shader is not None:
            with timing('Update shader visibility attribute'):  # TEMP_VISIBILITY_PROFILE
                self._visibility_shader.set_mask(mask_arr)
        self.signals.pointVisibilityChanged.emit()

    
    def _build_pipeline(
        self,
        renderer: vtk.vtkRenderer,
        out_actors: list[vtk.vtkProp],
        
        coarse_ratio: float = 0.01 # point ratio of coarse LOD
    ) -> None:
        
        # numpy data -> vtk data -> vtk points -> poly -> vertex glyph filter -> mapper
        points_np = self.data.points.astype("float32", copy=False)
        
        def _build_points (points_np: NDArray[np.float32]
                          )-> Tuple[vtk.vtkPolyData, vtk.vtkMapper]:
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(ns.numpy_to_vtk(points_np, deep=False))
            
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_points)
            
            vertex = vtk.vtkVertexGlyphFilter() # points -> drawable 'vertex' objects
            vertex.SetInputData(poly)
            # Build topology once. Attribute changes must not rerun glyph generation.
            vertex.Update()
            poly.SetVerts(vertex.GetOutput().GetVerts())
            
    
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            mapper.SetScalarModeToUsePointData()
            mapper.Update()
            return poly, mapper
        
        # build mappers
        self._coarse_stride = max(1, int(1 / coarse_ratio))
        self._poly, self._mapper_fine = _build_points(points_np)
        self._poly_coarse, self._mapper_coarse = _build_points(points_np[::self._coarse_stride])
        self._apply_clipping_planes()

        

        # actor
        # vtkLODActor renders through an internal actor that does not forward
        # shader properties. Select mappers on a regular actor instead.
        actor = vtk.vtkActor()
        actor.SetMapper(self._mapper_fine)
        self._lod = PointCloudLOD(renderer, actor, self._mapper_fine, self._mapper_coarse)
        actor.GetProperty().SetPointSize(self.point_size)
        
        out_actors.append(actor)
        self._main_actor = actor
        self._blend = PointCloudBlend(actor, [
            (self._poly, self._mapper_fine, 1),
            (self._poly_coarse, self._mapper_coarse, self._coarse_stride),
        ])

        self._visibility_shader = PointCloudVisibility(actor, self._blend.pipelines)
        self._visibility_shader.set_mask(self._visibility_mask)
        self._selection_shader.attach()
        self._color_mapping = GPUColorMapping(self)
        # BaseLayer.attach exposes the actor to rendering immediately after
        # this method returns. Initialize both LODs' color scalars and shader
        # inputs first: VTK only declares vertexColorVSOutput with scalars set.
        self.update()

    def _apply_clipping_planes(self) -> None:
        for mapper in (self._mapper_fine, self._mapper_coarse):
            if mapper is None:
                continue
            mapper.RemoveAllClippingPlanes()
            for plane in self._clipping_planes:
                mapper.AddClippingPlane(plane)

    def set_clipping_planes(self, planes: list[vtk.vtkPlane]) -> None:
        self._clipping_planes = list(planes)
        self._apply_clipping_planes()

    def clear_clipping_planes(self) -> None:
        self._clipping_planes = []
        self._apply_clipping_planes()


    @property
    def visible_inds(self):
        inds = np.arange(self.data.points.shape[0])
        if self._visibility_mask is None:
            return inds
        else:
            return inds[self._visibility_mask]
        
    def data_index_from_picked_id(self, sub_id: int) -> int:
        coarse = self._main_actor is not None and self._main_actor.GetMapper() is self._mapper_coarse
        stride = self._coarse_stride if coarse else 1
        return int(sub_id * stride)

    @timed('Prepare field display')  # TEMP_VISIBILITY_PROFILE
    def _field_display(self, field: FieldBase | None):
        """Resolve a field using the same rules for foreground and background."""
        if field is not None:
            field = self._color_overrides.get(field.name, field)
        ctf = scalar_range = None
        colors_np : Optional[NDArray[np.uint8]] = None # (N,3) colors
        scalars_np : Optional[NDArray[np.float32]] = None # (N,) scalar

        if field is not None:
            # Keep field arrays aligned with the full geometry; visibility is
            # applied by the shader and does not change the color-map range.
            data = field.data
            if field.field_type not in (FieldType.SEMANTIC, FieldType.INSTANCE):
                mapping = default_map(field)
                values = components(field, mapping)
                maps = mapping.channels if isinstance(mapping, VectorColorMap) else (mapping,)
                results = []
                for channel, values_channel in zip(maps, values):
                    lo, hi = resolve_limits(channel, finite_extrema(values_channel))
                    t = np.full(len(values_channel), .5) if lo == hi else (values_channel-lo)/(hi-lo)
                    output = sample_map(channel, np.nan_to_num(t))
                    output[~np.isfinite(values_channel)] = .6
                    results.append(output)
                rgb = np.column_stack(results) if isinstance(mapping, VectorColorMap) else results[0]
                return (np.clip(rgb*255, 0, 255).astype(np.uint8), None, None, None)
            
            if field.field_type == FieldType.SEMANTIC:
                assert isinstance(field, SemanticSegmentation), "Unmatching definition."
                colors_np = field.schema.get_color_array(field.data)
                            
            elif field.field_type == FieldType.INSTANCE:
                assert isinstance(field, InstanceSegmentation), "Unmatching definition."
                colors_np = field.get_color_array()

        if colors_np is not None and field is not None and field.field_type in (
            FieldType.SEMANTIC, FieldType.INSTANCE
        ):
            colors_np = colors_np[:, :3].copy()
            unassigned = data.reshape(-1) == -1
            colors_np[unassigned] = self.unassigned_point_color[:3]

        return colors_np, scalars_np, ctf, scalar_range

    @staticmethod
    def _display_rgb(display, count):
        colors, scalars, ctf, _ = display
        if colors is not None:
            return colors
        if scalars is not None and ctf is not None:
            if not scalars.size:
                return np.empty((0, 3), dtype=np.uint8)
            values = ns.numpy_to_vtk(np.ascontiguousarray(scalars), deep=False)
            return ns.vtk_to_numpy(ctf.MapScalars(values, vtk.VTK_COLOR_MODE_MAP_SCALARS, 0))[:, :3].copy()
        return np.tile((np.asarray(theme.DEFAULT_OBJ_COLOR) * 255).astype(np.uint8), (count, 1))

    @timed('Point cloud update')  # TEMP_VISIBILITY_PROFILE
    def update(self) -> None:
        names = self.data.field_names
        previous_field_name = self._active_field_name
        if self._active_field_name not in names:
            self._active_field_name = names[0] if names else None
        if self.background_field_name not in names:
            self.background_field_name = None
        if self._poly is None or self._mapper_fine is None:
            return
        assert self._blend is not None and self._main_actor is not None

        points = self.data.points
        self._blend.enabled = self.background_field_name is not None

        with timing('Replace VTK arrays (fine and coarse)'):  # TEMP_VISIBILITY_PROFILE
            topology_changed = False
            for poly, mapper, stride in self._blend.pipelines:
                poly.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(points[::stride]), deep=False))
                if poly.GetNumberOfVerts() != poly.GetNumberOfPoints():
                    vertex = vtk.vtkVertexGlyphFilter()
                    vertex.SetInputData(poly)
                    vertex.Update()
                    poly.SetVerts(vertex.GetOutput().GetVerts())
                    topology_changed = True
                mapper.SetScalarModeToUsePointData()
                poly.Modified()
            if topology_changed:
                self._visibility_shader.set_mask(self._visibility_mask)
        if topology_changed:
            self._selection_shader.update_mask()
        self._color_mapping.refresh()
        if previous_field_name != self._active_field_name:
            self.signals.activeFieldChanged.emit()
        self.signals.fieldDataChanged.emit()

    def preview_color_map(self, field, mapping):
        if self._color_mapping is not None:
            self._color_mapping.preview(field, mapping)

    def set_point_size(self, size: int) -> None:
        size = int(max(1, min(size, 50)))  # clamp
        if size == self.point_size:
            return
        self.point_size = size
        if self._main_actor is not None:
            self._main_actor.GetProperty().SetPointSize(size)

    def increase_point_size(self, step: int = 1) -> int:
        self.set_point_size(self.point_size + step)
        print(f'Set point size to {self.point_size}')
        return self.point_size
        

    def decrease_point_size(self, step: int = 1) -> int:
        self.set_point_size(self.point_size - step)
        print(f'Set point size to {self.point_size}')
        return self.point_size

    @property
    def id(self) -> str:
        return super().id
    
    @property
    def browser_name(self) -> str:
        if self._visibility_mask is not None:
            return f'{super().browser_name}; ({self._visibility_mask.sum():,} / {self.   data.points.shape[0]:,})'
        else:
            return super().browser_name

    @browser_name.setter
    def browser_name(self, browser_name: str) -> None:
        self._browser_name = browser_name
        
    @property
    def browser_sel_descr(self) -> str | None:
        if self.active_selection is None:
            return None
        else:
            return f"{self.active_selection.shape[0]:,} points"
        
    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float,float,float]:
        if self._poly is None:
            raise RuntimeError(f"Build the {self.__class__.__name__} first.")
        
        poly = self._poly_coarse if self._main_actor.GetMapper() is self._mapper_coarse else self._poly
        p = poly.GetPoint(int(sub_id))
        return float(p[0]), float(p[1]), float(p[2]) # FIXME: potential errors if the actor has a transform applied
        
    def detach(self) -> None:
        if self._lod is not None:
            self._lod.close()
            self._lod = None
        self._selection_shader.detach()
        return super().detach()
    
