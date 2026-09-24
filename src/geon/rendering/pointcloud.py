import vtk 
from vtk.util import numpy_support as ns # type: ignore

from PyQt6.QtCore import QTimer

import numpy as np
from numpy.typing import NDArray

from geon.data.pointcloud import (PointCloudData, FieldType, 
                                  SemanticSegmentation, InstanceSegmentation,
                                  FieldBase
                                  )

from geon.config import theme

from ..data.definitions import ColorMap
from ..util.common import blend_colors
from .base import BaseLayer
from .pointcloud_blend import PointCloudBlend
from .pointcloud_lod import PointCloudLOD
from .util import build_vtk_color_transfer_function
from .layer_registry import layer_for


from dataclasses import dataclass, field
from typing import Optional, Tuple
import math



class SelectionOverlay:
    SELECTION_BASE_COLOR = (1., 1., 1.)
    SELECTION_PULSE_COLOR = (1., 0.5, 0)
    SELECTION_TIMER_TICK = 50
    
    def __init__(self, layer: "PointCloudLayer") -> None:
        self.layer : PointCloudLayer = layer
        self.selection_timer: QTimer = QTimer()
        self.selection_actor: Optional[vtk.vtkActor] = None
        self.selection_phase: float = 0.0
        self.selection_timer.timeout.connect(lambda: self.update_pulse())
        
    def update_overlay(self)->None:
        
        selection = self.layer.active_selection
        renderer = self.layer.renderer
        
        all_pts = self.layer.data.points
        selection_pts = all_pts[self.layer.active_selection]
        if renderer is None:
            return
        if renderer is None or selection is None or selection.size == 0:
            # clean up
            if self.selection_actor is not None:
                renderer.RemoveActor(self.selection_actor)
                self.selection_actor = None
            self.selection_timer.stop()
            return
    
        # setup new overlay
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(selection_pts))
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        mapper.ScalarVisibilityOff()
        if self.selection_actor is None:
            self.selection_actor = vtk.vtkActor()
            pt_size = self.layer.point_size + 2
            self.selection_actor.GetProperty().SetPointSize(pt_size)
            renderer.AddActor(self.selection_actor)
        self.selection_actor.SetMapper(mapper)
        self.selection_actor.GetProperty().SetColor(*self.SELECTION_PULSE_COLOR)
        if not self.selection_timer.isActive():
            self.selection_phase = 0.0
            self.selection_timer.start(self.SELECTION_TIMER_TICK)
        
    def set_point_size(self, base_size: int) -> None:
        if self.selection_actor is None:
            return
        self.selection_actor.GetProperty().SetPointSize(base_size + 2)

            

    def update_pulse(self)->None:
        renderer = self.layer.renderer
        if self.selection_actor is None or renderer is None:
            return
        self.selection_phase += 0.1
        t = (math.sin(self.selection_phase*4) + 1) / 2
        self.selection_actor.GetProperty().SetOpacity(t)
        renderer.GetRenderWindow().Render()
    
        



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
        self._visibility_mask:      Optional[NDArray[np.bool_]] = None
        self._active_selection:     Optional[NDArray[np.int32]] = None
        self._selection_overlay:    SelectionOverlay = SelectionOverlay(self)
        
        self._mapper_fine:      Optional[vtk.vtkMapper] = None
        self._mapper_coarse:    Optional[vtk.vtkMapper] = None
        self._clipping_planes: list[vtk.vtkPlane] = []

        self._main_actor: Optional[vtk.vtkActor] = None

        
        self.browser_name = browser_name
        self.point_size: int = 2
        
        

        

    

        # index of currently displayed scalar in each vector field
        self._vf_active_index: dict[str, int] = {}
    
    @property
    def active_field(self) -> FieldBase | None:
        name = self.active_field_name
        if name is None:
            return None
        f = self.data.get_fields(name)
        if len(f) == 0:
            return None
        return f[0]
        
    def _populate_vf_active_index(self) -> None:
        for f_name in self.data.field_names:
            if f_name in self._vf_active_index.keys():
                self._vf_active_index[f_name] = \
                    min(self._vf_active_index[f_name], 
                        max(0, self.data[f_name].shape[-1] - 1))
            else:
                self._vf_active_index[f_name] = 0
    @property
    def vf_active_index (self) -> dict[str, int]:
        self._populate_vf_active_index()
        return self._vf_active_index

    @property
    def active_field_name(self) -> Optional[str]:
        return self._active_field_name
    

    @active_field_name.setter
    def active_field_name(self, field_name:str) -> None:
        assert field_name in self.data.field_names
        self._active_field_name = field_name


    @property
    def active_selection(self) -> Optional[NDArray[np.int32]]:
        return self._active_selection
    
    @active_selection.setter
    def active_selection(self, selection: Optional[NDArray[np.int32]]) -> None:
        if selection is not None:
            assert selection.max() < self.data.points.shape[0]
            assert selection.min() >= 0
        self._active_selection = selection
        self._selection_overlay.update_overlay()
        
        
        
    
    
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
        if self._blend is not None:
            self._blend.set_mix(self.foreground_mix)

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
        self.vf_active_index[field_name] = clamped
        self.update()
        return clamped

    def _init_visibility_mask(self)->None:
        if self.data.points.ndim < 2:
            raise Exception\
                (f"Point cloud holds invalid data: {self.data.points.shape}")
        self._visibility_mask = np.ones((self.data.points.shape[0]),dtype=bool)
    
    def _reset_visibility_mask(self)->None:
        self._visibility_mask = None

    def set_visibility_mask(self, mask: Optional[NDArray[np.bool_]]) -> None:
        if mask is None:
            self._visibility_mask = None
            self.update()
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
        
        self.update()

    
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
            
    
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex.GetOutputPort())
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

        self._init_visibility_mask()

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
        return self.visible_inds[::stride][sub_id]

    def _field_display(self, field: FieldBase | None):
        """Resolve a field using the same rules for foreground and background."""
        ctf = scalar_range = None
        colors_np : Optional[NDArray[np.uint8]] = None # (N,3) colors
        scalars_np : Optional[NDArray[np.float32]] = None # (N,) scalar

        def construct_default_cmap(scalars_np: np.ndarray, cmap_type:Optional[str] = None):

            smin, smax = (scalars_np.min(), scalars_np.max()) if scalars_np.size else (0.0, 1.0)
            c_pos = np.array([smin, smax])
            cmap = ColorMap.get_cmap(cmap_type,tuple(c_pos.tolist()))
            return cmap
        
        if field is not None:
            data_visible = field.data[self._visibility_mask] \
                if self._visibility_mask is not None else field.data
            
            if field.field_type == FieldType.COLOR:
                if data_visible.ndim != 2:
                    raise ValueError(f"Unexpected color field shape: {data_visible.shape}")
                if np.all(np.logical_and(0. <= data_visible, data_visible <= 1.)):
                    colors_np = (data_visible * 255).astype(np.uint8)
                elif np.all(np.logical_and(0 <= data_visible, data_visible <= 255)):
                    colors_np = data_visible.astype(np.uint8)
            
            elif field.field_type == FieldType.SCALAR or\
                    field.field_type == FieldType.INTENSITY:
                cmap_type = 'gray' if field.field_type == FieldType.INTENSITY else None
                scalars_np = np.asarray(data_visible, dtype=np.float32).reshape(-1)
                color_map = field.color_map or construct_default_cmap(scalars_np, cmap_type)
                ctf, scalar_range = build_vtk_color_transfer_function(color_map)
            
            elif field.field_type == FieldType.VECTOR or \
                    field.field_type == FieldType.NORMAL:
                ind = self.vf_active_index[field.name]
                ind = max(0, min(ind, data_visible.shape[1] - 1))
                scalars_np = data_visible[:,ind]
                color_map = field.color_map or construct_default_cmap(scalars_np)
                ctf, scalar_range = build_vtk_color_transfer_function(color_map)

            elif field.field_type == FieldType.SEMANTIC:
                assert isinstance(field, SemanticSegmentation), "Unmatching definition."
                colors_np = field.schema.get_color_array(field.data)
                if self._visibility_mask is not None:
                    colors_np = colors_np[self._visibility_mask] 
                            
            elif field.field_type == FieldType.INSTANCE:
                assert isinstance(field, InstanceSegmentation), "Unmatching definition."
                colors_np = field.get_color_array()
                if self._visibility_mask is not None:
                    colors_np = colors_np[self._visibility_mask] 

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

    def update(self) -> None:
        names = self.data.field_names
        if self._active_field_name not in names:
            self._active_field_name = names[0] if names else None
        if self.background_field_name not in names:
            self.background_field_name = None
        if self._poly is None or self._mapper_fine is None:
            return
        assert self._blend is not None and self._main_actor is not None

        points = self.data.points[self._visibility_mask] if self._visibility_mask is not None else self.data.points
        display = self._field_display(self.active_field)
        colors, scalars, ctf, scalar_range = display
        if self.background_field_name is not None:
            colors = self._display_rgb(display, len(points))
            background = self._field_display(self.data.get_fields(self.background_field_name)[0])
            self._blend.enable(self._display_rgb(background, len(points)), self.foreground_mix)
        else:
            self._blend.disable()

        for poly, mapper, stride in self._blend.pipelines:
            poly.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(points[::stride]), deep=False))
            mapper.SetScalarModeToUsePointData()
            if colors is not None:
                poly.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(colors[::stride]), deep=False))
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
                mapper.SetLookupTable(None)
            elif scalars is not None:
                poly.GetPointData().SetScalars(ns.numpy_to_vtk(np.ascontiguousarray(scalars[::stride]), deep=False))
                mapper.SetColorModeToMapScalars()
                mapper.ScalarVisibilityOn()
                mapper.SetLookupTable(ctf)
                mapper.SetUseLookupTableScalarRange(True)
                mapper.SetScalarRange(*scalar_range)
            else:
                poly.GetPointData().SetScalars(None)
                mapper.ScalarVisibilityOff()
                self._main_actor.GetProperty().SetColor(*theme.DEFAULT_OBJ_COLOR)
            poly.Modified()

    def set_point_size(self, size: int) -> None:
        size = int(max(1, min(size, 50)))  # clamp
        self.point_size = size
        if self._main_actor is not None:
            self._main_actor.GetProperty().SetPointSize(size)
        self._selection_overlay.set_point_size(size)
        self.update()

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
        self._selection_overlay.selection_timer.stop()
        sel_actor = self._selection_overlay.selection_actor
        if self.renderer is not None and sel_actor is not None:
            self.renderer.RemoveActor(sel_actor)
        return super().detach()
    
