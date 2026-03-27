import numpy as np
import pytest

from geon._native import region_merge
from geon.algorithms.region_merge import merge_planar_regions


def _adjacent_coplanar_labels():
    x_left = np.linspace(-0.05, -0.005, 20, dtype=np.float32)
    x_right = np.linspace(0.005, 0.05, 20, dtype=np.float32)
    y = np.linspace(-0.05, 0.05, 10, dtype=np.float32)
    xx_l, yy_l = np.meshgrid(x_left, y)
    xx_r, yy_r = np.meshgrid(x_right, y)
    left = np.stack([xx_l.ravel(), yy_l.ravel(), np.zeros(xx_l.size, dtype=np.float32)], axis=1)
    right = np.stack([xx_r.ravel(), yy_r.ravel(), np.zeros(xx_r.size, dtype=np.float32)], axis=1)
    coords = np.concatenate([left, right], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.zeros(left.shape[0], dtype=np.int32),
        np.ones(right.shape[0], dtype=np.int32),
    ])
    return coords, labels


def test_region_merge_shape_error():
    bad = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.int32)
    with pytest.raises(RuntimeError, match=r"coords must be a \(N,3\) float array"):
        region_merge.merge_planar_regions(bad, labels)


def test_adjacent_coplanar_regions_merge():
    coords, labels = _adjacent_coplanar_labels()
    merged, stats = region_merge.merge_planar_regions(
        coords,
        labels,
        {
            "neighbor_radius": 0.02,
            "min_contact_points": 5,
            "planarity_threshold": 0.6,
            "normal_angle_deg": 10.0,
            "plane_distance_threshold": 0.01,
            "min_region_size": 20,
        },
        None,
    )
    assert np.unique(merged[merged >= 0]).size == 1
    assert stats["num_output_regions"] == 1


def test_parallel_but_separated_regions_do_not_merge():
    coords, labels = _adjacent_coplanar_labels()
    coords[labels == 1, 0] += 1.0
    merged, stats = merge_planar_regions(
        coords,
        labels,
        params={
            "neighbor_radius": 0.02,
            "min_contact_points": 5,
            "planarity_threshold": 0.6,
            "normal_angle_deg": 10.0,
            "plane_distance_threshold": 0.01,
            "min_region_size": 20,
        },
    )
    assert np.unique(merged[merged >= 0]).size == 2
    assert stats["num_adjacency_pairs"] == 0


def test_large_normal_mismatch_does_not_merge():
    y = np.linspace(-0.05, 0.05, 10, dtype=np.float32)
    z = np.linspace(-0.05, 0.05, 10, dtype=np.float32)
    yy, zz = np.meshgrid(y, z)
    plane_a = np.stack([np.zeros(yy.size, dtype=np.float32), yy.ravel(), zz.ravel()], axis=1)
    plane_b = np.stack([np.full(yy.size, 0.01, dtype=np.float32), yy.ravel(), zz.ravel()], axis=1)
    plane_b[:, [0, 2]] = plane_b[:, [2, 0]]
    coords = np.concatenate([plane_a, plane_b], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.zeros(plane_a.shape[0], dtype=np.int32),
        np.ones(plane_b.shape[0], dtype=np.int32),
    ])
    merged, _ = merge_planar_regions(
        coords,
        labels,
        params={
            "neighbor_radius": 0.03,
            "min_contact_points": 5,
            "planarity_threshold": 0.6,
            "normal_angle_deg": 5.0,
            "plane_distance_threshold": 0.02,
            "min_region_size": 20,
        },
    )
    assert np.unique(merged[merged >= 0]).size == 2


def test_non_planar_regions_do_not_merge_and_minus_one_preserved():
    rng = np.random.default_rng(4)
    blob_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.02, size=(80, 3)).astype(np.float32)
    blob_b = rng.normal(loc=[0.03, 0.0, 0.0], scale=0.02, size=(80, 3)).astype(np.float32)
    coords = np.concatenate([blob_a, blob_b, np.array([[1.0, 1.0, 1.0]], dtype=np.float32)], axis=0)
    labels = np.concatenate([
        np.zeros(blob_a.shape[0], dtype=np.int32),
        np.ones(blob_b.shape[0], dtype=np.int32),
        np.array([-1], dtype=np.int32),
    ])
    merged, _ = merge_planar_regions(
        coords,
        labels,
        params={
            "neighbor_radius": 0.03,
            "min_contact_points": 5,
            "planarity_threshold": 0.9,
            "normal_angle_deg": 10.0,
            "plane_distance_threshold": 0.02,
            "min_region_size": 20,
        },
    )
    assert merged[-1] == -1
    assert np.unique(merged[:-1][merged[:-1] >= 0]).size == 2
