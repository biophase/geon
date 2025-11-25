# PointCloud object, fields, segmentation state 

from dataclasses import dataclass, field
from typing import Dict, Optional
from numpy.typing import NDArray
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
import vtk
from vtk.util import numpy_support as ns # type: ignore

@dataclass
class PointCloud:
    # main data containers
    points: np.ndarray = field(default_factory= lambda: np.empty(shape=(0,3),dtype=np.float32))
    colors: Optional[np.ndarray] = None
    fields: Dict[str, NDArray[np.float32]] = field(default_factory=lambda:{})

    poly_data: Optional[vtk.vtkPolyData] = None

    # utility
    def setupVtk(self):
        vtk_points = vtk.vtkPoints()
        array_view = ns.numpy_to_vtk(self.points, deep=False)
        vtk_points.SetData(array_view)
        self.poly_data = vtk.vtkPolyData()
        self.poly_data.SetPoints(vtk_points)
        


        

class AppState(QObject):
    # signals
    def __init__(self):
        super().__init__()
        

