import vtk 
from vtk.util import numpy_support as ns # type: ignore

from PyQt6.QtCore import QTimer

import numpy as np
from numpy.typing import NDArray

from geon.data.pointcloud import (PointCloudData, FieldType, 
                                  SemanticSegmentation, InstanceSegmentation,
                                  FieldBase
                                  )

from config import theme

from ..data.definitions import ColorMap
from ..util.common import blend_colors
from .base import BaseLayer
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
        self._visibility_mask:      Optional[NDArray[np.bool_]] = None
        self._active_selection:     Optional[NDArray[np.int32]] = None
        self._selection_overlay:    SelectionOverlay = SelectionOverlay(self)
        
        self._mapper_fine:      Optional[vtk.vtkMapper] = None
        self._mapper_coarse:    Optional[vtk.vtkMapper] = None

        self._main_actor: Optional[vtk.vtkLODActor] = None

        
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
                        self.data[f_name].shape[-1])
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
        self._poly, self._mapper_fine = _build_points(points_np)
        self._poly_coarse, self._mapper_coarse = _build_points(points_np[::int(1/coarse_ratio)])

        

        # actor
        actor = vtk.vtkLODActor()
        actor.SetMapper(self._mapper_fine)
        actor.AddLODMapper(self._mapper_coarse)
        actor.GetProperty().SetPointSize(self.point_size)
        
        out_actors.append(actor)
        self._main_actor = actor

        self._init_visibility_mask()


    @property
    def visible_inds(self):
        inds = np.arange(self.data.points.shape[0])
        if self._visibility_mask is None:
            return inds
        else:
            return inds[self._visibility_mask]
        
    def data_index_from_picked_id(self, sub_id: int) -> int:
        return self.visible_inds[sub_id]

    def update(self) -> None:
        ctf = None
        scalar_range = None

        if self._poly is None or self._mapper_fine is None:
            return

        # get displayed field
        if self._active_field_name is not None:
            field = self.data.get_fields(self._active_field_name)[0]
        elif self.data.field_num:
            field = self.data.get_fields(field_index=0)[0]
        else:
            field = None
        
        colors_np : Optional[NDArray[np.uint8]] = None # (N,3) colors
        scalars_np : Optional[NDArray[np.float32]] = None # (N,) scalar

        def construct_default_cmap(scalars_np: np.ndarray, cmap_type:Optional[str] = None):

            smin, smax = scalars_np.min(), scalars_np.max()
            c_pos = np.array([smin, smax])
            cmap = ColorMap.get_cmap(cmap_type,tuple(c_pos.tolist()))
            return cmap
        
        # get colors/scalars for active field
        # TODO: implement for coarse_mapper without code duplication
        if field is None:
            self._mapper_fine.ScalarVisibilityOff()
            if self._main_actor is not None:
                r,g,b, = theme.DEFAULT_OBJ_COLOR
                self._main_actor.GetProperty().SetColor(r,g,b)

        else:
            
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

        # geometry
        if self._visibility_mask is not None:
            visible_inds = np.nonzero(self._visibility_mask)[0]
            visible_points_np = self.data.points[visible_inds]

        else:
            visible_points_np = self.data.points

        vtk_points = self._poly.GetPoints()
        vtk_points.SetData(ns.numpy_to_vtk(visible_points_np, deep=False))
        
        # push scalars/colors to vtk
        if colors_np is not None:
            vtk_colors = ns.numpy_to_vtk(colors_np, deep=False)
            vtk_colors.SetNumberOfComponents(3)
            self._poly.GetPointData().SetScalars(vtk_colors)

            self._mapper_fine.SetScalarModeToUsePointData()
            self._mapper_fine.ScalarVisibilityOn()
            self._mapper_fine.SetLookupTable(None)
        elif scalars_np is not None:
            vtk_scalars = ns.numpy_to_vtk(scalars_np, deep=False)
            vtk_scalars.SetNumberOfComponents(1)
            self._poly.GetPointData().SetScalars(vtk_scalars)
            
            self._mapper_fine.SetScalarModeToUsePointData()
            self._mapper_fine.ScalarVisibilityOn()

            if ctf is not None and scalar_range is not None:
                # print(ctf)
                # print(scalar_range)
                # raise Exception
                self._mapper_fine.SetLookupTable(ctf)
                self._mapper_fine.SetUseLookupTableScalarRange(True)
                self._mapper_fine.SetScalarRange(*scalar_range)

        else:
            self._mapper_fine.ScalarVisibilityOff()
        self._poly.Modified()



    def set_point_size(self, size: int) -> None:
        size = int(max(1, min(size, 50)))  # clamp
        self.point_size = size
        if self._main_actor is not None:
            self._main_actor.GetProperty().SetPointSize(size)
        self._selection_overlay.set_point_size(size)
        # TODO:  also apply to any LOD actors that might be created internally
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
        
        p = self._poly.GetPoint(int(sub_id))
        return float(p[0]), float(p[1]), float(p[2]) # FIXME: potential errors if the actor has a transform applied
        
    def detach(self) -> None:
        sel_actor = self._selection_overlay.selection_actor
        if self.renderer is not None and sel_actor is not None:
            self.renderer.RemoveActor(sel_actor)
        return super().detach()
    
