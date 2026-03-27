from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import region_growing as _native
from ..data.pointcloud import PointCloudData

Progress = _native.Progress
NativeSeededGrower = _native.SeededGrower


def _as_coords(
    data_or_coords: PointCloudData | NDArray[np.float32],
) -> NDArray[np.float32]:
    if isinstance(data_or_coords, PointCloudData):
        coords = data_or_coords.points
    else:
        coords = data_or_coords
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be a (N,3) array, got {arr.shape}")
    return arr


def estimate_parameters(
    data_or_coords: PointCloudData | NDArray[np.float32],
    *,
    sample_size: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    coords = _as_coords(data_or_coords)
    epsilon, tau, alpha_deg, diagnostics = _native.estimate_parameters(
        coords,
        sample_size=int(sample_size),
        seed=int(seed),
    )
    return {
        "epsilon": float(epsilon),
        "tau": int(tau),
        "alpha_deg": float(alpha_deg),
        "diagnostics": diagnostics,
    }


def segment_planar_regions(
    data_or_coords: PointCloudData | NDArray[np.float32],
    *,
    normals: Optional[NDArray[np.float32]] = None,
    normal_mode: Literal["compute", "use_provided"] = "compute",
    params: Optional[dict[str, Any]] = None,
    chunking: Optional[dict[str, Any]] = None,
    merge: Optional[dict[str, Any]] = None,
    progress: Optional[Progress] = None,
) -> Tuple[NDArray[np.int32], dict[str, Any]]:
    coords = _as_coords(data_or_coords)
    normals_arr: NDArray[np.float32] | None = None
    if normals is not None:
        normals_arr = np.ascontiguousarray(np.asarray(normals, dtype=np.float32))
        if normals_arr.ndim != 2 or normals_arr.shape[1] != 3:
            raise ValueError(f"normals must be a (N,3) array, got {normals_arr.shape}")
        if normals_arr.shape[0] != coords.shape[0]:
            raise ValueError("normals must have same row count as coords")

    labels, stats = _native.segment_planar_regions(
        coords,
        normals_arr,
        normal_mode=normal_mode,
        params=params or {},
        chunking=chunking or {},
        merge=merge or {},
        progress=progress,
    )
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    return labels, stats


class SeededGrower:
    def __init__(
        self,
        data_or_coords: PointCloudData | NDArray[np.float32],
        *,
        normals: Optional[NDArray[np.float32]] = None,
        normal_mode: Literal["compute", "use_provided"] = "compute",
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        coords = _as_coords(data_or_coords)
        normals_arr: NDArray[np.float32] | None = None
        if normals is not None:
            normals_arr = np.ascontiguousarray(np.asarray(normals, dtype=np.float32))
            if normals_arr.ndim != 2 or normals_arr.shape[1] != 3:
                raise ValueError(f"normals must be a (N,3) array, got {normals_arr.shape}")
            if normals_arr.shape[0] != coords.shape[0]:
                raise ValueError("normals must have same row count as coords")
        self._native = NativeSeededGrower(
            coords,
            normals_arr,
            normal_mode=normal_mode,
            params=params or {},
        )

    def grow(self, seed_index: int) -> Tuple[NDArray[np.int32], dict[str, Any]]:
        indices, stats = self._native.grow(int(seed_index))
        return np.asarray(indices, dtype=np.int32).reshape(-1), stats
