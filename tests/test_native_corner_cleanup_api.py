import numpy as np
import pytest

from geon._native import corner_cleanup
from geon.algorithms.corner_cleanup import cleanup_corner_regions


def _two_planes_and_corner():
    axis = np.linspace(-0.05, 0.05, 11, dtype=np.float32)
    first, second = np.meshgrid(axis, axis)
    horizontal = np.stack(
        [first.ravel(), second.ravel(), np.zeros(first.size, dtype=np.float32)],
        axis=1,
    )
    vertical = np.stack(
        [np.zeros(first.size, dtype=np.float32), second.ravel(), first.ravel()],
        axis=1,
    )
    corner = np.stack(
        [
            np.where(axis < 0, 0.0004, 0.0008),
            axis,
            np.where(axis < 0, 0.0008, 0.0004),
        ],
        axis=1,
    ).astype(np.float32)
    coords = np.concatenate([horizontal, vertical, corner])
    labels = np.concatenate(
        [
            np.zeros(horizontal.shape[0], dtype=np.int32),
            np.ones(vertical.shape[0], dtype=np.int32),
            np.full(corner.shape[0], 2, dtype=np.int32),
        ]
    )
    return coords, labels, corner.shape[0]


def _params():
    return {
        "epsilon": 0.01,
        "neighbor_radius_factor": 2.0,
        "slenderness_threshold": 0.02,
        "absolute_width_factor": 3.0,
        "relative_width_factor": 0.5,
        "dual_support_threshold": 0.5,
        "min_corner_size": 5,
        "min_planar_size": 20,
    }


def test_corner_cleanup_shape_error():
    with pytest.raises(RuntimeError, match=r"coords must be a \(N,3\) float array"):
        corner_cleanup.cleanup_corner_regions(
            np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            np.asarray([0, 0, 0], dtype=np.int32),
        )


def test_elongated_region_between_two_planes_is_redistributed():
    coords, labels, corner_size = _two_planes_and_corner()

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert np.count_nonzero(cleaned == 2) == 0
    assert set(cleaned[-corner_size:].tolist()) == {0, 1}
    assert stats["num_corner_candidates"] == 1
    assert stats["num_corner_regions_cleaned"] == 1
    assert stats["num_corner_points_reassigned"] == corner_size
    assert stats["neighbor_radius"] == pytest.approx(0.02)


def test_long_narrow_planar_destinations_are_not_rejected_as_nonplanar():
    coords, labels, corner_size = _two_planes_and_corner()
    horizontal_count = 11 * 11
    vertical_count = 11 * 11
    coords[:horizontal_count, 0] *= 20.0
    coords[horizontal_count:horizontal_count + vertical_count, 2] *= 20.0

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert np.count_nonzero(cleaned == 2) == 0
    assert set(cleaned[-corner_size:].tolist()) == {0, 1}
    assert stats["num_destination_regions"] >= 2
    assert stats["num_corner_regions_cleaned"] == 1


def test_closest_jointly_bordering_pair_is_used_when_third_plane_is_in_radius():
    coords, labels, corner_size = _two_planes_and_corner()
    horizontal_count = 11 * 11
    vertical = coords[horizontal_count:2 * horizontal_count].copy()
    vertical[:, 0] = 0.019
    coords = np.concatenate([coords[:-corner_size], vertical, coords[-corner_size:]])
    labels = np.concatenate([
        labels[:-corner_size],
        np.full(vertical.shape[0], 3, dtype=np.int32),
        labels[-corner_size:],
    ])

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert np.count_nonzero(cleaned == 2) == 0
    assert set(cleaned[-corner_size:].tolist()) == {0, 1}
    assert stats["num_destination_regions"] >= 3
    assert stats["num_corner_regions_cleaned"] == 1


def test_coplanar_corner_is_split_by_nearest_region_points():
    axis = np.linspace(-0.1, 0.1, 21, dtype=np.float32)
    left_x, left_y = np.meshgrid(
        np.linspace(-0.05, -0.0012, 6, dtype=np.float32), axis
    )
    right_x, right_y = np.meshgrid(
        np.linspace(0.0012, 0.05, 6, dtype=np.float32), axis
    )
    left = np.stack(
        [left_x.ravel(), left_y.ravel(), np.zeros(left_x.size, dtype=np.float32)], axis=1
    )
    right = np.stack(
        [right_x.ravel(), right_y.ravel(), np.zeros(right_x.size, dtype=np.float32)], axis=1
    )
    corner = np.stack(
        [
            np.where(axis < 0, -0.0006, 0.0006),
            axis,
            np.zeros(axis.size, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    coords = np.concatenate([left, right, corner])
    labels = np.concatenate([
        np.zeros(left.shape[0], dtype=np.int32),
        np.ones(right.shape[0], dtype=np.int32),
        np.full(corner.shape[0], 2, dtype=np.int32),
    ])

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert set(cleaned[-corner.shape[0]:].tolist()) == {0, 1}
    assert stats["num_corner_regions_cleaned"] == 1


def test_corner_with_only_one_planar_neighbor_is_preserved():
    coords, labels, corner_size = _two_planes_and_corner()
    labels[labels == 1] = -1

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert np.all(cleaned[-corner_size:] == 2)
    assert stats["num_corner_candidates"] == 1
    assert stats["num_corner_regions_cleaned"] == 0
    assert stats["num_corner_points_reassigned"] == 0


def test_insufficient_dual_support_preserves_corner():
    coords, labels, corner_size = _two_planes_and_corner()
    params = _params()
    params["dual_support_threshold"] = 1.0
    coords[-1] += np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

    cleaned, stats = cleanup_corner_regions(coords, labels, params=params)

    assert np.all(cleaned[-corner_size:] == 2)
    assert stats["num_corner_regions_cleaned"] == 0


def test_absolute_width_gate_rejects_corner_wider_than_epsilon_scale():
    coords, labels, corner_size = _two_planes_and_corner()
    params = _params()
    params["epsilon"] = 1e-5
    params["neighbor_radius_factor"] = 2000.0

    cleaned, stats = cleanup_corner_regions(coords, labels, params=params)

    assert np.all(cleaned[-corner_size:] == 2)
    assert stats["num_rejected_absolute_width"] >= 1


def test_relative_width_gate_requires_wider_destination_regions():
    coords, labels, corner_size = _two_planes_and_corner()
    horizontal_count = 11 * 11
    coords[:horizontal_count, 0] *= 0.002
    coords[horizontal_count:2 * horizontal_count, 2] *= 0.002

    cleaned, stats = cleanup_corner_regions(coords, labels, params=_params())

    assert np.all(cleaned[-corner_size:] == 2)
    assert stats["num_rejected_relative_width"] >= 1
