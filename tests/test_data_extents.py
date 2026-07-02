import numpy as np

from geon.data.boundingbox import BoundingBox, BoundingBoxData
from geon.data.camera import CameraData
from geon.data.cellcomplex import CellComplexData, VertexCell
from geon.data.pointcloud import PointCloudData


def test_pointcloud_extents() -> None:
    data = PointCloudData(
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [-1.0, 4.0, 8.0],
            ],
            dtype=np.float32,
        )
    )

    assert data.get_extents() == (-1.0, 1.0, 2.0, 4.0, 3.0, 8.0)


def test_cellcomplex_extents() -> None:
    data = CellComplexData(
        vertices=[
            VertexCell(position=(0.0, -1.0, 2.0)),
            VertexCell(position=(3.0, 4.0, 5.0)),
        ]
    )

    assert data.get_extents() == (0.0, 3.0, -1.0, 4.0, 2.0, 5.0)


def test_bounding_box_data_extents_and_camera_none() -> None:
    data = BoundingBoxData(
        [
            BoundingBox(
                center_bottom_xyz=(1.0, 2.0, 3.0),
                width=2.0,
                depth=4.0,
                height=5.0,
            )
        ]
    )

    assert data.get_extents() == (0.0, 2.0, 0.0, 4.0, 3.0, 8.0)
    assert CameraData().get_extents() is None
