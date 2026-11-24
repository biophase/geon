import pytest
import vtk

from geon.ui.viewer import _fit_parallel_depth_range


def test_parallel_range_includes_geometry_behind_camera():
    source = vtk.vtkCubeSource()
    source.SetBounds(-2, 2, -2, 2, -10, 10)
    source.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddObserver(vtk.vtkCommand.StartEvent, _fit_parallel_depth_range)
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    camera.SetPosition(0, 0, 1)
    camera.SetFocalPoint(0, 0, 0)
    renderer.InvokeEvent(vtk.vtkCommand.StartEvent)
    near, far = camera.GetClippingRange()
    assert near < -9 and far > 11
    camera.SetPosition(0, 0, -20)
    camera.SetFocalPoint(0, 0, -21)
    renderer.InvokeEvent(vtk.vtkCommand.StartEvent)
    near, far = camera.GetClippingRange()
    assert near < -30 and far > -10
    camera.SetParallelProjection(False)
    camera.SetClippingRange(0.5, 100)
    renderer.InvokeEvent(vtk.vtkCommand.StartEvent)
    assert camera.GetClippingRange() == pytest.approx((0.5, 100))


def test_empty_parallel_scene_keeps_valid_range():
    renderer = vtk.vtkRenderer()
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    camera.SetClippingRange(0.1, 100)
    _fit_parallel_depth_range(renderer)
    assert camera.GetClippingRange() == pytest.approx((0.1, 100))
