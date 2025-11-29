import vtk 
from vtk.util import numpy_support as ns # type: ignore


from geon.data.pointcloud import PointCloudData
from .base import BaseLayer


class PointCloudLayer(BaseLayer[PointCloudData]):
    layer_type_id = "pointcloud"
    
    def _build_pipeline(
        self, 
        renderer: vtk.vtkRenderer, 
        out_actors: list[vtk.vtkProp]
    ) -> None:
        
        # numpy data -> vtk data -> vtk points -> poly -> vertex glyph filter -> mapper
        points_np = self.data.points.astype("float32", copy=False)
        
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(points_np, deep=False))
        
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        
        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        # which one:?
            # mapper.ScalarVisibilityOff
            # mapper.SetScalarModeToUsePointData()
        
# Here you can choose how to initialize colors: base colors, intensity, etc.
# For now: no scalars, just plain white actor.
vertex = vtk.vtkVertexGlyphFilter()
vertex.SetInputData(poly)