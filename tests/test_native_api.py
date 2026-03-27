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


def test_native_voxel_hash_neighbor_lookup_is_stable_at_problematic_grid_intervals():
    voxel_size = 0.02
    radius = 0.021

    # Regression for the old int->float->int voxel-key round-trip bug.
    # These indices used to alias to the previous bucket for some radii.
    xs = np.array([0.56, 0.58, 0.6001], dtype=np.float32)  # buckets 28, 29, 30
    coords = np.stack(
        [
            xs,
            np.zeros_like(xs, dtype=np.float32),
            np.zeros_like(xs, dtype=np.float32),
        ],
        axis=1,
    )

    voxel_hash = features.compute_voxel_hash(coords, inv_s=1.0 / voxel_size)
    neighbors = features.get_neighbor_inds_radius(
        radius=radius,
        query=coords[1],
        voxel_size=voxel_size,
        voxel_hash=voxel_hash,
        positive_coords=coords,
    )

    assert set(neighbors.tolist()) == {0, 1, 2}
