import vtk
from dataclasses import dataclass

from PyQt6.QtCore import QTimer


@dataclass
class RawPickResult:
    prop: vtk.vtkProp | None
    element_id: int
    association: str


class PointPicker:
    def __init__(self, renderer : vtk.vtkRenderer, radius_px: int = 1 ):
        self.renderer = renderer
        self.radius = int(radius_px)
        
        
        self._point_selector = vtk.vtkOpenGLHardwareSelector()
        self._point_selector.SetRenderer(self.renderer)
        self._point_selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)

        self._cell_selector = vtk.vtkOpenGLHardwareSelector()
        self._cell_selector.SetRenderer(self.renderer)
        self._cell_selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
        
        
    def pick(self, interactor: vtk.vtkRenderWindowInteractor, x: int, y: int):
        return self.pick_point(interactor, x, y)

    def pick_point(self, interactor: vtk.vtkRenderWindowInteractor, x: int, y: int):
        return self._pick_with_selector(interactor, x, y, self._point_selector, "point")

    def pick_cell(self, interactor: vtk.vtkRenderWindowInteractor, x: int, y: int):
        return self._pick_with_selector(interactor, x, y, self._cell_selector, "cell")

    def _pick_with_selector(
        self,
        interactor: vtk.vtkRenderWindowInteractor,
        x: int,
        y: int,
        selector: vtk.vtkOpenGLHardwareSelector,
        association: str,
    ):
        rw = interactor.GetRenderWindow()
        w, h = rw.GetSize()
        
        # y = h - 1 -y
        
        r = self.radius
        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(w - 1, x + r)
        y1 = min(h - 1, y + r)
        
        
        selector.SetArea(x0,y0,x1,y1)
        selection = selector.Select()
        if selection is None or selection.GetNumberOfNodes() == 0:
            return None
        
        # nodes should be depth-sorted, so picking closest node/actor here
        node = selection.GetNode(0)
        ids = node.GetSelectionList()
        if ids is None or ids.GetNumberOfTuples() == 0:
            return None
        
        element_id = int(ids.GetValue(0))
        props = node.GetProperties()
        prop_id = props.Get(vtk.vtkSelectionNode.PROP_ID()) if props else None
        picked_prop = selector.GetPropFromID(int(prop_id)) \
            if prop_id is not None else None
            
        return RawPickResult(picked_prop, element_id, association)
        
        
        
