import numpy as np
import pytest

from geon._native import superpoints
from geon.algorithms.superpoints import segment_superpoints
from geon.data.pointcloud import FieldType, PointCloudData


def _make_three_patch_cloud(n_per_patch: int = 60) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = np.stack(
        [
            rng.uniform(-1.0, -0.2, n_per_patch),
            rng.uniform(-1.0, -0.2, n_per_patch),
            np.zeros(n_per_patch),
        ],
        axis=1,
    )
    b = np.stack(
        [
            rng.uniform(0.2, 1.0, n_per_patch),
            rng.uniform(-1.0, -0.2, n_per_patch),
            np.zeros(n_per_patch),
        ],
        axis=1,
    )
    c = np.stack(
        [
            rng.uniform(-0.4, 0.4, n_per_patch),
            rng.uniform(0.2, 1.0, n_per_patch),
            np.ones(n_per_patch) * 0.5,
        ],
        axis=1,
    )
    return np.concatenate([a, b, c], axis=0).astype(np.float32)


def test_segment_superpoints_shape_error():
    bad = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match=r"coords must be a \(N,3\) float array"):
        superpoints.segment_superpoints(bad)


def test_segment_superpoints_returns_labels_and_stats():
    coords = _make_three_patch_cloud(40)
    labels, stats = superpoints.segment_superpoints(
        coords,
        k_neighbors=10,
        regularization=0.005,
        spatial_weight=0.2,
        cutoff=5,
        iterations=10,
        parallel=False,
        verbose=False,
    )
    assert labels.dtype == np.int32
    assert labels.ndim == 1
    assert labels.shape[0] == coords.shape[0]
    assert isinstance(stats, dict)
    assert stats["num_points"] == coords.shape[0]
    assert stats["num_superpoints"] >= 1
    assert stats["feature_dim"] == 3


def test_segment_superpoints_feature_shape_error():
    coords = _make_three_patch_cloud(30)
    bad_features = np.ones((coords.shape[0] - 1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="features must have the same number of rows as coords"):
        superpoints.segment_superpoints(coords, bad_features)


def test_geometry_plus_fields_wrapper_changes_partition():
    coords = _make_three_patch_cloud(40)
    pcd = PointCloudData(coords)
    field = np.zeros((coords.shape[0], 1), dtype=np.float32)
    field[coords[:, 0] > 0.0, 0] = 10.0
    pcd.add_field("feat", field, field_type=FieldType.SCALAR)

    labels_geom, _ = segment_superpoints(
        pcd,
        k_neighbors=10,
        regularization=0.005,
        spatial_weight=0.2,
        cutoff=5,
        iterations=8,
        parallel=False,
    )
    labels_feat, _ = segment_superpoints(
        pcd,
        feature_field_names=["feat"],
        k_neighbors=10,
        regularization=0.005,
        spatial_weight=0.2,
        cutoff=5,
        iterations=8,
        parallel=False,
    )
    assert labels_geom.shape == labels_feat.shape
    assert not np.array_equal(labels_geom, labels_feat)
