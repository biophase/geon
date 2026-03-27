from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import region_merge as _native

from ..data.pointcloud import PointCloudData

Progress = _native.Progress


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


def _as_labels(labels: NDArray[np.int32] | NDArray[np.int64]) -> NDArray[np.int32]:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(f"labels must have shape (N,) or (N,1), got {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"labels must have shape (N,) or (N,1), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int32)


def merge_planar_regions(
    data_or_coords: PointCloudData | NDArray[np.float32],
    labels: NDArray[np.int32] | NDArray[np.int64],
    *,
    params: Optional[dict[str, Any]] = None,
    progress: Optional[Progress] = None,
) -> Tuple[NDArray[np.int32], dict[str, Any]]:
    coords = _as_coords(data_or_coords)
    labels_arr = _as_labels(labels)
    if labels_arr.shape[0] != coords.shape[0]:
        raise ValueError("labels must have the same row count as coords")
    merged, stats = _native.merge_planar_regions(
        coords,
        labels_arr,
        params or {},
        progress,
    )
    merged = np.asarray(merged, dtype=np.int32).reshape(-1)
    return merged, stats
