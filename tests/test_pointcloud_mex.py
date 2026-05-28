import numpy as np
import pytest
import h5py

from geon.data.pointcloud import FieldType, InstanceSegmentation, PointCloudData, mex
from geon.rendering.pointcloud import PointCloudLayer


class _DummyViewer:
    def rerender(self) -> None:
        pass


class _DummyContext:
    def __init__(self) -> None:
        self.viewer = _DummyViewer()


def test_mex_accepts_column_vectors() -> None:
    data = np.asarray([[0], [1], [3]], dtype=np.int32)

    assert mex(data) == 2


def test_instance_segmentation_next_id_accepts_column_vectors() -> None:
    field = InstanceSegmentation(
        "instances",
        data=np.asarray([[0], [1], [3]], dtype=np.int32),
    )

    assert field.get_next_instance_id() == 2


def test_instance_segmentation_normalizes_flat_data_to_column_vector() -> None:
    field = InstanceSegmentation(
        "instances",
        data=np.asarray([0, 1, 3], dtype=np.int32),
    )

    assert field.data.shape == (3, 1)
    assert field.data[:, 0].tolist() == [0, 1, 3]


def test_instance_segmentation_size_uses_column_vector() -> None:
    field = InstanceSegmentation("instances", size=3)

    assert field.data.shape == (3, 1)
    assert field.data[:, 0].tolist() == [0, 0, 0]


def test_instance_segmentation_hdf5_load_migrates_flat_data(tmp_path) -> None:
    path = tmp_path / "instances.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("instances")
        group.attrs["field_type"] = FieldType.INSTANCE.name
        group.create_dataset("data", data=np.asarray([0, 1, 3], dtype=np.int32))

    with h5py.File(path, "r") as h5:
        field = InstanceSegmentation.from_hdf5_fieldgroup(h5["instances"])

    assert field.data.shape == (3, 1)
    assert field.get_next_instance_id() == 2


def test_annotate_points_assigns_instance_column_vector_and_undoes() -> None:
    annotate = pytest.importorskip("geon.tools.annotate", exc_type=ImportError)
    pcd = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    pcd.add_field(
        name="instances",
        data=np.asarray([0, 1, 3, 3], dtype=np.int32),
        field_type=FieldType.INSTANCE,
    )
    layer = PointCloudLayer(pcd)
    layer.active_selection = np.asarray([1, 2], dtype=np.int32)
    field = pcd.get_fields(names="instances")[0]
    ctx = _DummyContext()
    cmd = annotate.AnnotatePointsCmd(
        title="Annotate points",
        sem_field_name=None,
        inst_field_name="instances",
        sem_inds_old=None,
        inst_inds_old=None,
        sem_ind_new=None,
        layer_ref=lambda: layer,
        ctx_ref=lambda: ctx,
    )

    cmd.execute()

    assert field.data.shape == (4, 1)
    assert field.data[:, 0].tolist() == [0, 2, 2, 3]

    cmd.undo()

    assert field.data[:, 0].tolist() == [0, 1, 3, 3]


def test_mex_rejects_non_flat_2d_arrays() -> None:
    data = np.asarray([[0, 1], [2, 3]], dtype=np.int32)

    with pytest.raises(AssertionError, match="flat arrays"):
        mex(data)
