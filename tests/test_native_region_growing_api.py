import numpy as np
import pytest

from geon._native import region_growing


def _make_two_plane_cloud(n_per_plane: int = 120) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, n_per_plane).astype(np.float32)
    y = rng.uniform(-2.0, 2.0, n_per_plane).astype(np.float32)
    z0 = np.zeros(n_per_plane, dtype=np.float32)
    z1 = np.full(n_per_plane, 1.0, dtype=np.float32)
    plane_a = np.stack([x, y, z0], axis=1)
    plane_b = np.stack([x, y, z1], axis=1)
    return np.concatenate([plane_a, plane_b], axis=0).astype(np.float32)


def test_estimate_parameter_shape_error():
    bad = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match=r"coords must be a \(N,3\) float array"):
        region_growing.estimate_parameters(bad, sample_size=100, seed=0)


def test_estimate_parameter_deterministic_seed():
    coords = _make_two_plane_cloud(80)
    est_a = region_growing.estimate_parameters(coords, sample_size=120, seed=7)
    est_b = region_growing.estimate_parameters(coords, sample_size=120, seed=7)
    assert pytest.approx(est_a[0], rel=1e-6, abs=1e-6) == est_b[0]  # epsilon
    assert int(est_a[1]) == int(est_b[1])  # tau
    assert pytest.approx(est_a[2], rel=1e-6, abs=1e-6) == est_b[2]  # alpha_deg


def test_segment_planar_regions_returns_labels_and_stats():
    coords = _make_two_plane_cloud(120)
    labels, stats = region_growing.segment_planar_regions(
        coords,
        normal_mode="compute",
        params={
            "epsilon": 0.25,
            "tau": 10,
            "alpha_deg": 35.0,
            "confidence": 0.99,
            "perform_cca": True,
            "verbose": False,
        },
        chunking={
            "enabled": True,
            "mode": "explicit",
            "chunk_x": 2,
            "chunk_y": 1,
            "chunk_z": 1,
            "overlap_factor": 3.0,
        },
        merge={
            "angle_deg": 5.0,
            "distance_factor": 3.0,
        },
    )
    assert labels.dtype == np.int32
    assert labels.ndim == 1
    assert labels.shape[0] == coords.shape[0]
    assert isinstance(stats, dict)
    assert stats["num_points"] == coords.shape[0]
    assert "num_regions_post_merge" in stats

    unique = np.unique(labels[labels >= 0])
    assert unique.size >= 2


def test_segment_planar_regions_accepts_provided_normals():
    coords = _make_two_plane_cloud(100)
    normals = np.zeros_like(coords, dtype=np.float32)
    normals[:, 2] = 1.0
    labels, _ = region_growing.segment_planar_regions(
        coords,
        normals,
        normal_mode="use_provided",
        params={
            "epsilon": 0.25,
            "tau": 8,
            "alpha_deg": 35.0,
            "confidence": 0.99,
            "perform_cca": True,
            "verbose": False,
        },
        chunking={"enabled": True, "mode": "auto", "target_points_per_chunk": 100},
        merge={"angle_deg": 5.0, "distance_factor": 3.0},
    )
    assert labels.shape[0] == coords.shape[0]
