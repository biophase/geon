import numpy as np
import pytest

from geon._native import plane_ransac


def _make_two_plane_cloud(n_per_plane: int = 120) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, n_per_plane).astype(np.float32)
    y = rng.uniform(-2.0, 2.0, n_per_plane).astype(np.float32)
    z0 = np.zeros(n_per_plane, dtype=np.float32)
    z1 = np.full(n_per_plane, 1.0, dtype=np.float32)
    plane_a = np.stack([x, y, z0], axis=1)
    plane_b = np.stack([x, y, z1], axis=1)
    return np.concatenate([plane_a, plane_b], axis=0).astype(np.float32)


def test_segment_planes_shape_error():
    bad = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match=r"coords must be a \(N,3\) float array"):
        plane_ransac.segment_planes(bad)


def test_segment_planes_returns_labels_and_stats():
    coords = _make_two_plane_cloud(100)
    labels, stats = plane_ransac.segment_planes(
        coords,
        normal_mode="compute",
        params={
            "epsilon": 0.15,
            "min_points": 30,
            "normal_threshold_deg": 35.0,
            "cluster_epsilon": 0.2,
            "probability": 0.01,
            "max_iterations_per_plane": 1000,
            "seed": 4,
        },
    )
    assert labels.dtype == np.int32
    assert labels.ndim == 1
    assert labels.shape[0] == coords.shape[0]
    assert isinstance(stats, dict)
    assert stats["num_points"] == coords.shape[0]
    assert stats["num_planes"] >= 2


def test_segment_planes_accepts_provided_normals():
    coords = _make_two_plane_cloud(90)
    normals = np.zeros_like(coords, dtype=np.float32)
    normals[:, 2] = 1.0
    labels, _ = plane_ransac.segment_planes(
        coords,
        normals,
        normal_mode="use_provided",
        params={
            "epsilon": 0.15,
            "min_points": 20,
            "normal_threshold_deg": 35.0,
            "cluster_epsilon": 0.2,
            "probability": 0.01,
            "max_iterations_per_plane": 1000,
            "seed": 1,
        },
    )
    assert labels.shape[0] == coords.shape[0]


def test_random_scatter_stays_mostly_unassigned():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    labels, stats = plane_ransac.segment_planes(
        coords,
        normal_mode="compute",
        params={
            "epsilon": 0.05,
            "min_points": 40,
            "normal_threshold_deg": 20.0,
            "cluster_epsilon": 0.05,
            "probability": 0.01,
            "max_iterations_per_plane": 500,
            "seed": 2,
        },
    )
    assert np.count_nonzero(labels >= 0) <= 40
    assert stats["num_unassigned"] >= 160


def test_disconnected_coplanar_patches_become_distinct_planes():
    rng = np.random.default_rng(7)
    x0 = rng.uniform(-1.0, -0.5, 60).astype(np.float32)
    y0 = rng.uniform(-1.0, -0.5, 60).astype(np.float32)
    z0 = np.zeros(60, dtype=np.float32)
    x1 = rng.uniform(0.5, 1.0, 60).astype(np.float32)
    y1 = rng.uniform(0.5, 1.0, 60).astype(np.float32)
    z1 = np.zeros(60, dtype=np.float32)
    coords = np.concatenate(
        [
            np.stack([x0, y0, z0], axis=1),
            np.stack([x1, y1, z1], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    normals = np.zeros_like(coords, dtype=np.float32)
    normals[:, 2] = 1.0
    labels, stats = plane_ransac.segment_planes(
        coords,
        normals,
        normal_mode="use_provided",
        params={
            "epsilon": 0.05,
            "min_points": 20,
            "normal_threshold_deg": 25.0,
            "cluster_epsilon": 0.05,
            "probability": 0.01,
            "max_iterations_per_plane": 1000,
            "seed": 9,
        },
    )
    unique = np.unique(labels[labels >= 0])
    assert unique.size >= 2
    assert stats["num_planes"] >= 2
