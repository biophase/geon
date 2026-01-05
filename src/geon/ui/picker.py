import vtk

from PyQt6.QtCore import QTimer


class PointPicker:
    def __init__(self, renderer : vtk.vtkRenderer, radius_px: int = 1 ):
        self.renderer = renderer
        self.radius = int(radius_px)
        
        
        self._selector = vtk.vtkOpenGLHardwareSelector()
        self._selector.SetRenderer(self.renderer)
        self._selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
        
        
    def pick(self, interactor: vtk.vtkRenderWindowInteractor, x: int, y: int):
        rw = interactor.GetRenderWindow()
        w, h = rw.GetSize()
        
        # y = h - 1 -y
        
        r = self.radius
        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(w - 1, x + r)
        y1 = min(h - 1, y + r)
        
        
        self._selector.SetArea(x0,y0,x1,y1)
        selection = self._selector.Select()
        if selection is None or selection.GetNumberOfNodes() == 0:
            return None
        
        # nodes should be depth-sorted, so picking closest node/actor here
        node = selection.GetNode(0)
        ids = node.GetSelectionList()
        if ids is None or ids.GetNumberOfTuples() == 0:
            return None
        
        point_id = int(ids.GetValue(0))
        props = node.GetProperties()
        prop_id = props.Get(vtk.vtkSelectionNode.PROP_ID()) if props else None
        picked_prop = self._selector.GetPropFromID(int(prop_id)) \
            if prop_id is not None else None
            
        return picked_prop, point_id
        
        
        