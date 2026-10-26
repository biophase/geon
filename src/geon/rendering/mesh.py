import numpy as np
import vtk
from vtk.util import numpy_support as ns

from geon.config.theme import DEFAULT_OBJ_COLOR
from geon.data.mesh import MeshData
from .base import BaseLayer
from .layer_registry import layer_for


@layer_for(MeshData)
class MeshLayer(BaseLayer[MeshData]):
    layer_type_id = "mesh"
    use_cell_picking_for_selection = True

    def __init__(self, data, browser_name=None):
        super().__init__(data)
        self.browser_name = browser_name or data.name
        self._poly = None
        self._mapper = None
        self._actor = None

    def _build_pipeline(self, renderer, out_actors):
        self._poly = vtk.vtkPolyData()
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputData(self._poly)
        self._mapper.ScalarVisibilityOff()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().SetColor(*DEFAULT_OBJ_COLOR[:3])
        self._actor.SetVisibility(self.visible)
        self.update()
        out_actors.append(self._actor)

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

    def data_index_from_picked_id(self, sub_id):
        if not 0 <= sub_id < len(self.data.vertices):
            raise IndexError(sub_id)
        return int(sub_id)

    def world_xyz_from_picked_id(self, sub_id):
        return tuple(float(v) for v in self.data.vertices[self.data_index_from_picked_id(sub_id)])

    def data_index_from_pick(self, sub_id, prop, association):
        if association == "cell":
            if not 0 <= sub_id < len(self.data.faces):
                raise IndexError(sub_id)
            return int(sub_id)
        return self.data_index_from_picked_id(sub_id)

    def world_xyz_from_pick(self, sub_id, prop, association):
        index = self.data_index_from_pick(sub_id, prop, association)
        if association == "cell":
            return tuple(float(v) for v in self.data.vertices[self.data.faces[index]].mean(axis=0))
        return self.world_xyz_from_picked_id(index)
