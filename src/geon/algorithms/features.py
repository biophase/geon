from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from geon._native import features as _native
from ..data.pointcloud import PointCloudData, FieldType

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
    optional_feature_field_names: Optional[dict[str, Optional[str]]] = None,
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
    # Avoid NaNs/inf from native computation or downstream ratios.
    eigenvalues = np.nan_to_num(eigenvalues, nan=0.0, posinf=0.0, neginf=0.0)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    if progress is not None and progress.cancelled():
        return
    if field_name_normals is None:
        field_name_normals = f'normals(r={radius:.3f})'
    if compute_normals:
        data.add_field(field_name_normals, np.abs(normals), FieldType.NORMAL)
    
    if field_name_eigenvals is None:
        field_name_eigenvals = f'eigenvalues(r={radius:.3f})'
    if compute_eigenvals:
        data.add_field(field_name_eigenvals, eigenvalues, FieldType.VECTOR, vector_dim_hint=3)

    selected_optional = optional_feature_field_names or {}
    if not selected_optional:
        return

    l1 = eigenvalues[:, 0]
    l2 = eigenvalues[:, 1]
    l3 = eigenvalues[:, 2]
    l_sum = l1 + l2 + l3

    with np.errstate(divide="ignore", invalid="ignore"):
        feature_values: dict[str, NDArray[np.float32]] = {
            "sum_eigenvalues": l_sum,
            "omnivariance": np.cbrt(np.clip(l1 * l2 * l3, a_min=0.0, a_max=None)),
            "eigenentropy": -(
                np.where(l1 > 0.0, l1 * np.log(l1), 0.0)
                + np.where(l2 > 0.0, l2 * np.log(l2), 0.0)
                + np.where(l3 > 0.0, l3 * np.log(l3), 0.0)
            ),
            "anisotropy": (l3 - l1) / l3,
            "planarity": (l2 - l1) / l3,
            "linearity": (l3 - l2) / l3,
            "pca1": l3 / l_sum,
            "pca2": l2 / l_sum,
            "surface_variation": l1 / l_sum,
            "sphericity": l1 / l3,
            "verticality": 1.0 - np.abs(normals[:, 2]),
            "eigenvalue_1": l1,
            "eigenvalue_2": l2,
            "eigenvalue_3": l3,
        }

    for key, custom_name in selected_optional.items():
        values = feature_values.get(key)
        if values is None:
            continue
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        field_name = custom_name if custom_name else f"{key}(r={radius:.3f})"
        data.add_field(field_name, values[:, None], FieldType.SCALAR)


