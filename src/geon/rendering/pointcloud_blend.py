"""Fine/coarse pipelines shared by mapping, visibility, and selection."""

import vtk


class PointCloudBlend:
    ARRAY = "geonBackgroundInput"

    def __init__(self, actor: vtk.vtkActor,
                 pipelines: list[tuple[vtk.vtkPolyData, vtk.vtkPolyDataMapper, int]]) -> None:
        self.actor = actor
        self.pipelines = pipelines
        self.enabled = False

