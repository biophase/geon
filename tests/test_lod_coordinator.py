from dataclasses import dataclass

import numpy as np
import pytest
import vtk

from geon.data.pointcloud import PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.lod import LODCoordinator


@dataclass
class Participant:
    estimates: list
    visible: bool = True
    level: int = 0
    samples: int = 0

    def costs(self):
        return self.estimates

    def select_level(self, level):
        self.level = level

    def record_frame(self):
        self.samples += 1
        return self.estimates[self.level]


@pytest.mark.parametrize('costs,budget,previous,expected', [
    ([[.045, .008], [.012, .003]], .057, [0, 0], [0, 0]),
    ([[.070, .008], [.012, .003]], .057, [0, 0], [1, 0]),
    ([[.070, .008], [.012, .003]], .010, [0, 0], [1, 1]),
    ([[.045, .008]], .050, [1], [1]),  # Restoration needs headroom.
    ([[.035, .008]], .050, [1], [0]),
    ([[.010, .010]], .001, [0], [0]),  # No saving: preserve detail.
    ([[.070, .008]], 0, [0], [1]),  # Best effort, never hide the object.
])
def test_global_budget(costs, budget, previous, expected):
    assert LODCoordinator.choose_levels(costs, budget, previous) == expected


def test_non_lod_props_do_not_divide_budget_and_hidden_clouds_are_excluded():
    renderer = vtk.vtkRenderer()
    coordinator = LODCoordinator.for_renderer(renderer)
    cloud = Participant([.020, .002])
    hidden = Participant([1, .1], visible=False)
    coordinator.register(cloud)
    coordinator.register(hidden)
    for index in range(10):
        actor = vtk.vtkActor()
        actor.SetVisibility(index % 2)
        renderer.AddActor(actor)
    renderer.SetAllocatedRenderTime(.067)
    coordinator.overhead = .010
    coordinator._start()
    assert cloud.level == 0
    coordinator.overhead = .060
    coordinator._start()
    assert cloud.level == 1
    coordinator.unregister(cloud)
    coordinator.unregister(hidden)
    assert not renderer.HasObserver(vtk.vtkCommand.StartEvent)
    assert not renderer.HasObserver(vtk.vtkCommand.EndEvent)


def test_shared_overhead_and_no_sampling_during_picking(monkeypatch):
    renderer = vtk.vtkRenderer()
    coordinator = LODCoordinator.for_renderer(renderer)
    cloud = Participant([.020, .002])
    coordinator.register(cloud)
    clock = iter([10.0, 10.04])
    monkeypatch.setattr('geon.rendering.lod.perf_counter', lambda: next(clock))
    coordinator._start()
    coordinator._end()
    assert cloud.samples == 1
    assert coordinator.overhead == pytest.approx(.004)
    selector = vtk.vtkOpenGLHardwareSelector()
    monkeypatch.setattr(renderer, 'GetSelector', lambda: selector)
    cloud.level = 1
    coordinator._start()
    coordinator._end()
    assert cloud.level == 0
    assert cloud.samples == 1
    assert coordinator.overhead == pytest.approx(.004)
    monkeypatch.setattr(renderer, 'GetSelector', lambda: None)
    coordinator.unregister(cloud)


def test_idle_restores_fine_even_with_tight_budget():
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    coordinator = LODCoordinator.for_renderer(renderer)
    cloud = Participant([.100, .010])
    coordinator.register(cloud)
    renderer.SetAllocatedRenderTime(.020)
    window.SetDesiredUpdateRate(interactor.GetDesiredUpdateRate())
    coordinator._start()
    assert cloud.level == 1
    window.SetDesiredUpdateRate(interactor.GetStillUpdateRate())
    coordinator._start()
    assert cloud.level == 0
    coordinator.unregister(cloud)
    window.Finalize()


def test_clouds_share_coordinator_and_detach_independently():
    renderer = vtk.vtkRenderer()
    layers = [PointCloudLayer(PointCloudData(np.zeros((201, 3), dtype=np.float32)))
              for _ in range(2)]
    for layer in layers:
        layer.attach(renderer)
        layer.update()
    coordinator = layers[0]._lod.coordinator
    assert coordinator is layers[1]._lod.coordinator
    layers[0]._lod._costs = [.070, .008]
    layers[1]._lod._costs = [.012, .003]
    renderer.SetAllocatedRenderTime(.067)
    coordinator.overhead = .010
    coordinator._start()
    assert layers[0]._main_actor.GetMapper() is layers[0]._mapper_coarse
    assert layers[1]._main_actor.GetMapper() is layers[1]._mapper_fine
    layers[0].detach()
    assert len(coordinator.participants) == 1
    layers[1].detach()
    assert not hasattr(renderer, '_geon_lod_coordinator')
    layers[0].attach(renderer)
    assert layers[0]._lod.coordinator is not coordinator
    layers[0].detach()


def test_pick_scope_protects_depth_prepass_and_recovers_from_failure():
    renderer = vtk.vtkRenderer()
    coordinator = LODCoordinator.for_renderer(renderer)
    cloud = Participant([.100, .010])
    coordinator.register(cloud)
    renderer.SetAllocatedRenderTime(.020)
    coordinator._start()
    assert cloud.level == 1
    with pytest.raises(RuntimeError):
        with coordinator.picking():
            assert renderer.GetSelector() is None  # Preparatory depth pass.
            coordinator._start()
            coordinator._end()
            assert cloud.level == 0
            assert cloud.samples == 0
            raise RuntimeError('selection failed')
    assert coordinator._picking_depth == 0
    coordinator._start()
    assert cloud.level == 1
    coordinator.unregister(cloud)
