import numpy as np
import pytest

from geon._native import features


def test_compute_voxel_hash_shape_error():
    bad_coords = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="positive_coords must be a \\(N,3\\) float array"):
        features.compute_voxel_hash(bad_coords, inv_s=1.0)

