from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import corner_cleanup as _native

if TYPE_CHECKING:
    from ..data.pointcloud import PointCloudData

Progress = _native.Progress


def cleanup_corner_regions(
    data_or_coords: "PointCloudData" | NDArray[np.float32],
    labels: NDArray[np.int32] | NDArray[np.int64],
    *,
    params: Optional[dict[str, Any]] = None,
    progress: Optional[Progress] = None,
) -> Tuple[NDArray[np.int32], dict[str, Any]]:
    coords = data_or_coords.points if hasattr(data_or_coords, "points") else data_or_coords
    coords_array = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError(f"coords must be a (N,3) array, got {coords_array.shape}")

    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim == 2:
        if label_array.shape[1] != 1:
            raise ValueError(f"labels must have shape (N,) or (N,1), got {label_array.shape}")
        label_array = label_array[:, 0]
    if label_array.ndim != 1:
        raise ValueError(f"labels must have shape (N,) or (N,1), got {label_array.shape}")
    if label_array.shape[0] != coords_array.shape[0]:
        raise ValueError("labels must have the same row count as coords")

    cleaned, stats = _native.cleanup_corner_regions(
        coords_array,
        np.ascontiguousarray(label_array),
        params or {},
        progress,
    )
    return np.asarray(cleaned, dtype=np.int32).reshape(-1), stats
