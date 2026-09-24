"""Keep one shader-capable actor while selecting a point-cloud mapper."""

import vtk


class PointCloudLOD:
    def __init__(self, renderer: vtk.vtkRenderer, actor: vtk.vtkActor,
                 fine: vtk.vtkPolyDataMapper, coarse: vtk.vtkPolyDataMapper) -> None:
        self.renderer = renderer
        self.actor = actor
        self.fine = fine
        self.coarse = coarse
        self.observer = renderer.AddObserver(vtk.vtkCommand.StartEvent, self._select)

    def _select(self, *_) -> None:
        # As with vtkLODActor, use the measured full-resolution draw time and
        # the renderer's frame budget. Still renders have a generous budget.
        count = max(1, self.renderer.GetViewProps().GetNumberOfItems())
        budget = self.renderer.GetAllocatedRenderTime() / count
        mapper = self.coarse if self.fine.GetTimeToDraw() > budget else self.fine
        if self.actor.GetMapper() is not mapper:
            self.actor.SetMapper(mapper)

    def close(self) -> None:
        self.renderer.RemoveObserver(self.observer)
