import numpy as np
import pytest

from geon.algorithms.split_between import split_between


def _parallel_planes():
    axis = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    set_a = np.stack([x.ravel(), y.ravel(), np.zeros(x.size, np.float32)], axis=1)
    set_b = set_a.copy()
    set_b[:, 2] = 1.0
    work = np.asarray([[0, 0, 0.1], [0, 0, 0.9], [0, 0, 0.5]], np.float32)
    coords = np.concatenate([set_a, set_b, work])
    a = np.arange(set_a.shape[0], dtype=np.int64)
    b = np.arange(set_a.shape[0], set_a.shape[0] + set_b.shape[0], dtype=np.int64)
    working = np.arange(coords.shape[0] - work.shape[0], coords.shape[0], dtype=np.int64)
    return coords, working, a, b


def test_plane_distance_assignment_and_tie_to_a():
    coords, working, set_a, set_b = _parallel_planes()
    goes_to_b, stats = split_between(
        coords, working, set_a, set_b, "plane_distance")

    assert goes_to_b.tolist() == [False, True, False]
    assert stats["assigned_to_a"] == 2
    assert stats["assigned_to_b"] == 1


def test_nearest_neighbor_assignment_and_tie_to_a():
    coords, working, set_a, set_b = _parallel_planes()
    goes_to_b, _ = split_between(
        coords, working, set_a, set_b, "nearest_neighbor")

    assert goes_to_b.tolist() == [False, True, False]


def test_split_between_rejects_overlapping_sets():
    coords, working, set_a, set_b = _parallel_planes()
    with pytest.raises(RuntimeError, match="disjoint"):
        split_between(coords, working, set_a, set_a, "nearest_neighbor")


def test_plane_distance_rejects_collinear_reference_points():
    coords, working, set_a, set_b = _parallel_planes()
    set_a = set_a[:3]
    coords[set_a] = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    with pytest.raises(RuntimeError, match="non-collinear"):
        split_between(coords, working, set_a, set_b, "plane_distance")
