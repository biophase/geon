import vtk 
from vtk.util import numpy_support as ns # type: ignore
import numpy as np
from numpy.typing import NDArray

from geon.data.pointcloud import PointCloudData, FieldType, SemanticSegmentation, InstanceSegmentation
from geon.data.definitions import ColorMap
from config import theme
from .base import BaseLayer
from .util import build_vtk_color_transfer_function

from dataclasses import dataclass, field
from typing import Optional, Tuple



class PointCloudLayer(BaseLayer[PointCloudData]):
    layer_type_id = "pointcloud"

    def __init__(self, data:PointCloudData):
        super().__init__(data)
        self._poly:         Optional[vtk.vtkPolyData] = None
        self._poly_coarse:  Optional[vtk.vtkPolyData] = None

        self._active_field_name:    Optional[str] = None
        self._visibility_mask:      Optional[NDArray[np.bool_]] = None

        # index of currently displayed scalar in each vector field
        self._vf_active_index: dict[str, int] = {}
        
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

    def _init_visibility_mask(self)->None:
        if self.data.points.ndim < 2:
            raise Exception\
                (f"Point cloud holds invalid data: {self.data.points.shape}")
        self._visibility_mask = np.ones((self.data.points.shape[0]),dtype=bool)
    
    def _reset_visibility_mask(self)->None:
        self._visibility_mask = None

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
            
            vertex = vtk.vtkVertexGlyphFilter()
            vertex.SetInputData(poly)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex.GetOutputPort())
            mapper.SetScalarModeToUsePointData()
            mapper.Update()
            return poly, mapper
        
        # build mappers
        self._poly, mapper_fine = _build_points(points_np)
        self._poly_coarse, mapper_coarse = _build_points(points_np[::int(1/coarse_ratio)])

        # actor
        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper_fine)
        actor.AddLODMapper(mapper_coarse)

        out_actors.append(actor)
        self._init_visibility_mask()

        # further methods from old code:
        # self.renderer.RemoveAllViewProps()
        # self.renderer.AddActor(actor)
        # self.renderer.ResetCamera()
        # self.set_pivot_point(initial_pivot)
        # self.update_interactor()
        # self.vtkWidget.GetRenderWindow().Render()
        # # Apply initial colors (which will use the visibility mask)
        # self.update_point_cloud_actor_colors(self.all_colors)



    def update(self) -> None:
        ctf = None

        # get displayed field
        if self._active_field_name is not None:
            field = self.data.get_fields(self._active_field_name)[0]
        elif self.data.field_num:
            field = self.data.get_fields(field_index=0)[0]
        else:
            field = None
        
        # get colors
        if field is not None:
            
            data_visible = field.data[self._visibility_mask] \
                if self._visibility_mask is not None else field.data
            
            if field.field_type == FieldType.COLOR:
                if np.all((0. <= data_visible) & (data_visible <= 1.)):
                    colors_np = np.astype(data_visible*255, np.uint8)
                elif np.all((0 <= data_visible) & (data_visible <= 255)):
                    colors_np = data_visible
            
            elif field.field_type == FieldType.SCALAR:
                colors_np = data_visible
                color_map = field.color_map or ColorMap('default')
                ctf = build_vtk_color_transfer_function(color_map)
            
            elif field.field_type == FieldType.VECTOR:
                ind = self.vf_active_index[field.name]
                colors_np = data_visible[:,ind]
                color_map = field.color_map or ColorMap('default')
                ctf = build_vtk_color_transfer_function(color_map)

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

        if self._visibility_mask is not None:
            visible_inds = np.nonzero(self._visibility_mask)[0]
            visible_points_np = self.data.points[visible_inds]

        else:
            visible_points_np = self.data.points




    @property
    def name(self) -> str:
        return super().name
    
    @property
    def browser_name(self) -> str:
        if self._visibility_mask is not None:
            return f'self.name; (visible:{self._visibility_mask.sum()}/{self.   data.points.shape[0]})'
        else:
            return self.name
        

# # Here you can choose how to initialize colors: base colors, intensity, etc.
# # For now: no scalars, just plain white actor.
# vertex = vtk.vtkVertexGlyphFilter()
# vertex.SetInputData(poly)