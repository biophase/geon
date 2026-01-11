from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import features as _native

VoxelHash = _native.VoxelHash


def voxel_key(x: float, y: float, z: float, inv_s: float) -> int:
    return _native.voxel_key(x, y, z, inv_s)


def compute_voxel_hash(
    positive_coords: NDArray[np.float32],
    inv_s: float,
) -> VoxelHash:
    return _native.compute_voxel_hash(positive_coords, inv_s)


def get_neighbor_inds_radius(
    radius: float,
    query: NDArray[np.float32],
    voxel_size: float,
    voxel_hash: VoxelHash,
    positive_coords: NDArray[np.float32],
) -> NDArray[np.uint32]:
    return _native.get_neighbor_inds_radius(
        radius,
        query,
        voxel_size,
        voxel_hash,
        positive_coords,
    )


def compute_pcd_features(
    radius: float,
    voxel_size: float,
    positive_coords: NDArray[np.float32],
    voxel_hash: VoxelHash,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    return _native.compute_pcd_features(
        radius,
        voxel_size,
        positive_coords,
        voxel_hash,
    )
