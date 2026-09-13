import numpy as np
import pytest
import h5py

from geon.data.pointcloud import (
    FieldBase,
    FieldType,
    InstanceSegmentation,
    PointCloudData,
    SemanticSchema,
    SemanticSegmentation,
    mex,
)
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


def test_pointcloud_add_instance_field_accepts_flat_data() -> None:
    pcd = PointCloudData(np.zeros((3, 3), dtype=np.float32))

    pcd.add_field(
        name="instances",
        data=np.asarray([0, 1, 3], dtype=np.int32),
        field_type=FieldType.INSTANCE,
    )

    field = pcd.get_fields(names="instances")[0]
    assert field.data.shape == (3, 1)
    assert field.data[:, 0].tolist() == [0, 1, 3]


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


def test_extract_subcloud_copies_requested_fields_and_returns_source_indices() -> None:
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    pcd = PointCloudData(points)
    pcd.add_field(
        name="normals",
        data=np.arange(15, dtype=np.float32).reshape(5, 3),
        field_type=FieldType.NORMAL,
    )
    pcd.add_field(
        name="unused",
        data=np.arange(5, dtype=np.float32)[:, None],
        field_type=FieldType.SCALAR,
    )

    subcloud, source_indices = pcd.extract_subcloud(
        np.asarray([3, 1], dtype=np.int32),
        field_names=["normals"],
    )

    assert source_indices.dtype == np.int64
    assert source_indices.tolist() == [3, 1]
    np.testing.assert_array_equal(subcloud.points, points[[3, 1]])
    assert subcloud.field_names == ["normals"]
    normals = subcloud.get_fields(names="normals")[0]
    assert isinstance(normals, FieldBase)
    assert normals.field_type is FieldType.NORMAL
    np.testing.assert_array_equal(normals.data, pcd["normals"][[3, 1]])

    subcloud.points[0, 0] = -100
    normals.data[0, 0] = -100
    assert pcd.points[3, 0] != -100
    assert pcd["normals"][3, 0] != -100


def test_extract_subcloud_accepts_boolean_mask_and_no_fields_by_default() -> None:
    pcd = PointCloudData(np.arange(12, dtype=np.float32).reshape(4, 3))
    pcd.add_field(name="instances", data=np.arange(4), field_type=FieldType.INSTANCE)

    subcloud, source_indices = pcd.extract_subcloud(
        np.asarray([True, False, True, False]),
    )

    assert source_indices.tolist() == [0, 2]
    assert subcloud.field_names == []
    np.testing.assert_array_equal(subcloud.points, pcd.points[[0, 2]])


def test_extract_subcloud_preserves_specialized_field_types_and_metadata() -> None:
    pcd = PointCloudData(np.zeros((3, 3), dtype=np.float32))
    schema = SemanticSchema(name="classes")
    pcd.add_field(
        name="semantic",
        data=np.asarray([-1, 0, 0], dtype=np.int32),
        field_type=FieldType.SEMANTIC,
        schema=schema,
    )
    pcd.add_field(
        name="instances",
        data=np.asarray([0, 1, 1], dtype=np.int32),
        field_type=FieldType.INSTANCE,
    )

    subcloud, _ = pcd.extract_subcloud(
        np.asarray([2, 0]),
        field_names=["semantic", "instances"],
    )

    semantic = subcloud.get_fields(names="semantic")[0]
    instances = subcloud.get_fields(names="instances")[0]
    assert isinstance(semantic, SemanticSegmentation)
    assert isinstance(instances, InstanceSegmentation)
    assert semantic.schema.name == "classes"
    assert semantic.schema is not schema
    assert semantic.data.reshape(-1).tolist() == [0, -1]
    assert instances.data[:, 0].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        (np.asarray([[0, 1]]), ValueError),
        (np.asarray([True, False]), ValueError),
        (np.asarray([0, 0]), ValueError),
        (np.asarray([-1, 2]), IndexError),
        (np.asarray([0.0, 1.0]), TypeError),
    ],
)
def test_extract_subcloud_rejects_invalid_selection(selection, error) -> None:
    pcd = PointCloudData(np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(error):
        pcd.extract_subcloud(selection)


def test_extract_subcloud_rejects_missing_or_duplicate_fields() -> None:
    pcd = PointCloudData(np.zeros((3, 3), dtype=np.float32))
    pcd.add_field(name="normals", data=np.zeros((3, 3)), field_type=FieldType.NORMAL)

    with pytest.raises(KeyError, match="missing"):
        pcd.extract_subcloud(np.asarray([0]), field_names=["missing"])
    with pytest.raises(ValueError, match="duplicates"):
        pcd.extract_subcloud(np.asarray([0]), field_names=["normals", "normals"])


def test_instance_segmentation_maps_multiple_labels_to_distinct_free_ids() -> None:
    field = InstanceSegmentation("instances", np.asarray([0, 1, 3, 7]))

    mapping = field.map_to_free([5, 2, 9])

    assert mapping == {5: 2, 2: 4, 9: 5}
    assert field.data[:, 0].tolist() == [0, 1, 3, 7]

    with pytest.raises(ValueError, match="duplicates"):
        field.map_to_free([1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        field.map_to_free([-1])
    with pytest.raises(TypeError, match="integers"):
        field.map_to_free([1.5])  # type: ignore[list-item]


def test_instance_segmentation_merges_subset_and_preserves_unassigned() -> None:
    field = InstanceSegmentation("instances", np.asarray([0, 1, 3, 3, 7]))

    mapping = field.merge_with_segmentation(
        np.asarray([0, -1, 1], dtype=np.int32),
        inverse_mapping=np.asarray([1, 2, 4], dtype=np.int64),
    )

    assert mapping == {0: 2, 1: 4}
    assert field.data[:, 0].tolist() == [0, 2, 3, 3, 4]


def test_instance_segmentation_merges_full_field() -> None:
    field = InstanceSegmentation("instances", np.asarray([0, 2, 2]))

    mapping = field.merge_with_segmentation(np.asarray([-1, 0, 1]))

    assert mapping == {0: 1, 1: 3}
    assert field.data[:, 0].tolist() == [0, 1, 3]


def test_instance_segmentation_merge_validates_inputs() -> None:
    field = InstanceSegmentation("instances", size=3)

    with pytest.raises(ValueError, match="label count"):
        field.merge_with_segmentation(np.asarray([0, 1]))
    with pytest.raises(ValueError, match="length"):
        field.merge_with_segmentation(np.asarray([0]), np.asarray([0, 1]))
    with pytest.raises(IndexError, match="out-of-range"):
        field.merge_with_segmentation(np.asarray([0]), np.asarray([3]))
    with pytest.raises(ValueError, match="duplicate"):
        field.merge_with_segmentation(np.asarray([0, 1]), np.asarray([1, 1]))
    with pytest.raises(TypeError, match="integers"):
        field.merge_with_segmentation(np.asarray([0.5, 1.5, 2.5]))
