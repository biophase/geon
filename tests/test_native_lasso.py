import numpy as np
import pytest

from geon._native.lasso import select_polygon
from geon.tools.lasso import LassoTool


def reference(points, ids, matrix, polygon, keep_inside):
    xy = LassoTool._compute_display_coordinates(points[ids], matrix, 800, 600)
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    bbox = np.all((xy >= lo) & (xy <= hi), axis=1)
    inside = np.zeros(len(ids), dtype=bool)
    inside[bbox] = LassoTool.points_in_poly_(xy[bbox], polygon)
    return ids[inside if keep_inside else ~inside]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('perspective', [False, True])
@pytest.mark.parametrize('keep_inside', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_matches_numpy(dtype, perspective, keep_inside, strided):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(20000, 3)).astype(dtype)
    if strided:
        points = points[::2, ::-1]
    ids = rng.permutation(len(points)).astype(np.int64)[::2]
    # Include duplicate indices to check ordering and multiplicity preservation.
    ids = np.concatenate((ids, ids[:5]))
    matrix = np.array([[0.8, 0.1, 0.2, 0.1],
                       [-0.1, 0.9, 0.1, -0.2],
                       [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    if perspective:
        matrix[3] = [0.1, 0.2, 1, 0]
    # Concave polygon, repeated closing vertex, and horizontal edges.
    polygon = np.array([[100, 100], [700, 100], [400, 300],
                        [700, 500], [100, 500], [100, 100]])
    actual = select_polygon(points, ids, matrix, 800, 600, polygon, keep_inside)
    np.testing.assert_array_equal(actual, reference(points, ids, matrix, polygon, keep_inside))
    assert actual.dtype == np.int64


@pytest.mark.parametrize('keep_inside', [False, True])
def test_boundaries_nonfinite_and_zero_w(keep_inside):
    points = np.array([[-0.5, -0.5, 1], [0.5, 0.5, 1], [0, 0, 1],
                       [0, 0, -1], [0, 0, 0], [np.nan, 0, 1],
                       [np.inf, 0, 1]], dtype=np.float64)
    ids = np.arange(len(points))
    matrix = np.eye(4)
    matrix[3] = [0, 0, 1, 0]
    polygon = np.array([[200, 150], [600, 150], [600, 450], [200, 450]])
    with np.errstate(invalid='ignore'):
        expected = reference(points, ids, matrix, polygon, keep_inside)
    np.testing.assert_array_equal(
        select_polygon(points, ids, matrix, 800, 600, polygon, keep_inside), expected)


def test_empty_candidates_and_invalid_inputs():
    points = np.zeros((3, 3), dtype=np.float32)
    polygon = np.array([[0, 0], [800, 0], [400, 600]])
    assert select_polygon(points, np.array([], dtype=np.int64), np.eye(4),
                          800, 600, polygon).size == 0
    for ids in ([-1], [3]):
        with pytest.raises(ValueError, match='out of bounds'):
            select_polygon(points, np.array(ids), np.eye(4), 800, 600, polygon)
    with pytest.raises(ValueError, match='matrix'):
        select_polygon(points, np.array([0]), np.eye(3), 800, 600, polygon)
    with pytest.raises(ValueError, match='polygon'):
        select_polygon(points, np.array([0]), np.eye(4), 800, 600, polygon[:2])
    with pytest.raises(TypeError):
        select_polygon(points.astype(np.int32), np.array([0]), np.eye(4), 800, 600, polygon)
