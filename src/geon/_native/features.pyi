from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class VoxelHash:
    def __len__(self) -> int: ...


class Progress:
    def reset(self, total: int) -> None: ...
    def request_cancel(self) -> None: ...
    def cancelled(self) -> bool: ...
    def done(self) -> int: ...
    def total(self) -> int: ...


def voxel_key(x: float, y: float, z: float, inv_s: float) -> int: ...


def compute_voxel_hash(
    positive_coords: NDArray[np.float32],
    inv_s: float,
) -> VoxelHash: ...


def get_neighbor_inds_radius(
    radius: float,
    query: NDArray[np.float32],
    voxel_size: float,
    voxel_hash: VoxelHash,
    positive_coords: NDArray[np.float32],
) -> NDArray[np.uint32]: ...


def compute_pcd_features(
    radius: float,
    voxel_size: float,
    positive_coords: NDArray[np.float32],
    voxel_hash: VoxelHash,
    progress: Progress | None = ...,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]: ...
