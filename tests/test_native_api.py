import numpy as np

from geon._native import features


def test_native_voxel_hash_and_neighbors():
    coords = np.array(
        [
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
            [10.0, 10.0, 11.0],
        ],
        dtype=np.float32,
    )
    voxel_size = 1.0
    voxel_hash = features.compute_voxel_hash(coords, inv_s=1.0 / voxel_size)
    assert len(voxel_hash) > 0

    neighbors = features.get_neighbor_inds_radius(
        radius=1.5,
        query=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        voxel_size=voxel_size,
        voxel_hash=voxel_hash,
        positive_coords=coords,
    )
    assert neighbors.dtype == np.uint32
    assert neighbors.ndim == 1
    assert neighbors.size >= 1
    assert 0 in neighbors
