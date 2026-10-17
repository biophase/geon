import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as ns

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer


@pytest.mark.parametrize('coarse', [False, True])
def test_hidden_foreground_does_not_intercept_pick(monkeypatch, coarse):
    monkeypatch.setenv('GEON_SHADER_VISIBILITY', '1')
    # Coarse sample indices 0 and 100 overlap, with 0 closest to camera.
    points = np.full((101, 3), 100., dtype=np.float32)
    points[0] = [0, 0, 1]
    points[100] = [0, 0, 0]
    data = PointCloudData(points)
    data.add_field('color', np.tile([255, 0, 0], (101, 1)), FieldType.COLOR)
    layer = PointCloudLayer(data)
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(0)
    window.SetSize(100, 100)
    window.AddRenderer(renderer)
    layer.attach(renderer)
    layer.update()
    layer._lod.close()  # Force the same mapper for display and selection.
    layer._main_actor.SetMapper(layer._mapper_coarse if coarse else layer._mapper_fine)
    layer._main_actor.GetProperty().SetPointSize(15)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 5)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2)
    camera.SetClippingRange(.1, 10)
    selector = vtk.vtkOpenGLHardwareSelector()
    selector.SetRenderer(renderer)
    selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)

    def pick(x=50):
        selector.SetArea(x, 50, x, 50)
        selection = selector.Select()
        if selection is None or selection.GetNumberOfNodes() == 0:
            return None
        ids = ns.vtk_to_numpy(selection.GetNode(0).GetSelectionList())
        assert len(ids) == 1
        return layer.data_index_from_picked_id(int(ids[0]))

    def pixel():
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        return ns.vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(100, 100, -1)[50, 50, :3]

    try:
        before = layer._poly.GetPoints().GetMTime(), layer._poly.GetVerts().GetMTime()
        assert pick() == 0
        footprint = [pick(x) for x in range(43, 58)]
        mask = np.zeros(101, dtype=bool)
        mask[100] = True
        layer.set_visibility_mask(mask)
        assert pick() == 100
        # Hardware selection uses its own point footprint. Preserve that behavior.
        assert [pick(x) for x in range(43, 58)] == [
            100 if point_id is not None else None for point_id in footprint]
        assert pixel()[0] > 200
        mask = np.zeros(101, dtype=bool)
        layer.set_visibility_mask(mask)
        assert pick() is None
        np.testing.assert_array_equal(pixel(), [0, 0, 0])
        layer.set_visibility_mask(None)
        assert pick() == 0
        # Restore a pre-existing subset, as Accept/Cancel do when already isolated.
        mask[100] = True
        layer.set_visibility_mask(mask)
        assert pick() == 100
        layer.set_visibility_mask(None)
        assert before == (layer._poly.GetPoints().GetMTime(), layer._poly.GetVerts().GetMTime())
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, .5)
        plane.SetNormal(0, 0, -1)
        layer.set_clipping_planes([plane])
        assert pick() == 100
    finally:
        layer.detach()
        window.Finalize()


def test_geometry_count_change_rebuilds_topology(monkeypatch):
    monkeypatch.setenv('GEON_SHADER_VISIBILITY', '1')
    data = PointCloudData(np.zeros((1, 3), dtype=np.float32))
    layer = PointCloudLayer(data)
    layer.attach(vtk.vtkRenderer())
    data.points = np.zeros((201, 3), dtype=np.float32)
    layer.update()
    for poly, _, stride in layer._blend.pipelines:
        count = len(data.points[::stride])
        assert poly.GetNumberOfVerts() == count
        assert poly.GetPointData().GetArray('geonVisible').GetNumberOfTuples() == count
    layer.detach()
