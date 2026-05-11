import numpy as np
import pytest

from geon._native import subsampling


def test_spatial_subsample_mask_returns_bool_mask():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    mask = subsampling.spatial_subsample_mask(coords, min_distance=1.0)

    assert mask.dtype == np.bool_
    assert mask.shape == (4,)
    assert mask.tolist() == [True, False, True, True]


def test_spatial_subsample_mask_rejects_bad_shape():
    bad_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="coords must be a \\(N,3\\) float array"):
        subsampling.spatial_subsample_mask(bad_coords, min_distance=1.0)


def test_spatial_subsample_mask_rejects_non_positive_distance():
    coords = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(RuntimeError, match="min_distance must be > 0"):
        subsampling.spatial_subsample_mask(coords, min_distance=0.0)
