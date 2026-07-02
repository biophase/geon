import numpy as np
import vtk

from geon.data.pointcloud import PointCloudData
from geon.rendering.pointcloud import PointCloudLayer


def test_pointcloud_clipping_planes_do_not_change_visibility_mask() -> None:
    layer = PointCloudLayer(PointCloudData(np.zeros((4, 3), dtype=np.float32)))
    layer.attach(vtk.vtkRenderer())
    original_mask = layer._visibility_mask

    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(1.0, 0.0, 0.0)
    layer.set_clipping_planes([plane])

    assert layer._visibility_mask is original_mask
    assert layer._mapper_fine is not None
    assert layer._mapper_fine.GetNumberOfClippingPlanes() == 1

    layer.clear_clipping_planes()

    assert layer._visibility_mask is original_mask
    assert layer._mapper_fine.GetNumberOfClippingPlanes() == 0
