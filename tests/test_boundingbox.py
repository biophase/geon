from __future__ import annotations

import math

import h5py
import numpy as np

from geon.data.boundingbox import BoundingBox, BoundingBoxData
from geon.data.pointcloud import SemanticClass, SemanticSchema


def test_horizontal_box_uses_bottom_center_and_ccw_yaw() -> None:
    box = BoundingBox.from_horizontal_corners(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        1.0,
        5.0,
    )

    assert box.center_bottom_xyz == (1.0, 1.5, 1.0)
    assert box.dimensions == (2.0, 3.0, 4.0)
    assert box.yaw == 0.0
    corners = box.corners()
    assert np.allclose(corners[0], (0.0, 0.0, 1.0))
    assert np.allclose(corners[6], (2.0, 3.0, 5.0))


def test_horizontal_box_yaw_is_measured_from_positive_x_ccw() -> None:
    box = BoundingBox.from_horizontal_corners(
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (-3.0, 0.0, 0.0),
        0.0,
        1.0,
    )

    assert math.isclose(box.yaw, math.pi / 2.0)
    assert box.dimensions == (2.0, 3.0, 1.0)


def test_adjust_face_keeps_opposite_face_fixed() -> None:
    box = BoundingBox(
        center_bottom_xyz=(0.0, 0.0, 0.0),
        width=2.0,
        depth=4.0,
        height=6.0,
    )
    old_xmin = box.face_center("xmin").copy()

    box.adjust_face("xmax", 2.0)

    assert math.isclose(box.width, 4.0)
    assert np.allclose(box.face_center("xmin"), old_xmin)
    assert np.allclose(box.center_bottom_xyz, (1.0, 0.0, 0.0))


def test_bounding_box_data_hdf5_round_trip(tmp_path) -> None:
    schema = SemanticSchema(
        name="objects",
        semantic_classes=[SemanticClass(-1, "_unlabeled", (204, 204, 204)), SemanticClass(1, "chair", (1, 2, 3))],
    )
    data = BoundingBoxData(
        [
            BoundingBox(
                id="box_a",
                center_bottom_xyz=(1.0, 2.0, 3.0),
                yaw=0.25,
                pitch=0.1,
                roll=0.2,
                width=4.0,
                depth=5.0,
                height=6.0,
                semantic_id=1,
                attributes={"source": "test"},
            )
        ],
        schema=schema,
    )
    path = tmp_path / "bbox.h5"
    with h5py.File(path, "w") as handle:
        data.save_hdf5(handle.create_group("bbox"))
    with h5py.File(path, "r") as handle:
        loaded = BoundingBoxData.load_hdf5(handle["bbox"])

    assert loaded.schema is not None
    assert loaded.schema.name == "objects"
    assert loaded.box_count == 1
    box = loaded.boxes[0]
    assert box.id == "box_a"
    assert box.center_bottom_xyz == (1.0, 2.0, 3.0)
    assert box.semantic_id == 1
    assert box.attributes == {"source": "test"}
