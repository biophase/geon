"""Point-cloud detail levels for the shared viewport LOD coordinator."""

import vtk

from .lod import LODCoordinator


class PointCloudLOD:
    def __init__(self, renderer: vtk.vtkRenderer, actor: vtk.vtkActor,
                 fine: vtk.vtkPolyDataMapper, coarse: vtk.vtkPolyDataMapper) -> None:
        self.renderer = renderer
        self.actor = actor
        self.fine = fine
        self.coarse = coarse
        self.level = 0
        self._costs = [None, None]
        self.coordinator = LODCoordinator.for_renderer(renderer)
        self.coordinator.register(self)

    @property
    def visible(self):
        return bool(self.actor.GetVisibility())

    def costs(self):
        fine = self._costs[0] or 0.0
        coarse = self._costs[1]
        if coarse is None:
            fine_count = max(1, self.fine.GetInput().GetNumberOfPoints())
            ratio = self.coarse.GetInput().GetNumberOfPoints() / fine_count
            coarse = fine * ratio
        return [fine, coarse]

    def select_level(self, level):
        self.level = level
        mapper = self.fine if level == 0 else self.coarse
        if self.actor.GetMapper() is not mapper:
            self.actor.SetMapper(mapper)

    def record_frame(self):
        mapper = self.fine if self.level == 0 else self.coarse
        measured = max(0.0, mapper.GetTimeToDraw())
        old = self._costs[self.level]
        self._costs[self.level] = measured if old is None else 0.7 * old + 0.3 * measured
        return measured

    def invalidate(self):
        self._costs = [None, None]
        self.coordinator.overhead = 0.0

    def close(self) -> None:
        self.coordinator.unregister(self)
