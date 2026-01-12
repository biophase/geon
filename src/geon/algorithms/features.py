from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import features as _native
from ..data.pointcloud import PointCloudData, FieldType



from typing import Optional


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
    # voxel_size: float,
    # positive_coords: NDArray[np.float32],
    data: PointCloudData,
    field_name_normals: Optional[str]=None,
    field_name_eigenvals: Optional[str]=None,
    compute_normals: bool = True,
    compute_eigenvals: bool = True,
    progress: Optional[_native.Progress] = None,
    # voxel_hash: VoxelHash,
) -> None:
    
    coords = data.points
    positive_coords = coords - coords.min(axis=0)
    voxel_size = radius
    voxel_hash = compute_voxel_hash(positive_coords, inv_s=1/voxel_size)

    eigenvalues, normals= _native.compute_pcd_features(
        radius,
        voxel_size,
        positive_coords,
        voxel_hash,
        progress,
    )
    if progress is not None and progress.cancelled():
        return
    if field_name_normals is None:
        field_name_normals = f'normals(r={radius:.3f})'
    if compute_normals:
        data.add_field(field_name_normals, normals, FieldType.NORMAL)
    
    if field_name_eigenvals is None:
        field_name_eigenvals = f'eigenvalues(r={radius:.3f})'
    if compute_eigenvals:
        data.add_field(field_name_eigenvals, eigenvalues, FieldType.VECTOR, vector_dim_hint=3)


