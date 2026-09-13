from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import region_growing as _native

if TYPE_CHECKING:
    from ..data.pointcloud import PointCloudData


def segment_connected_components(
    data_or_coords: "PointCloudData" | NDArray[np.float32],
    *,
    epsilon: float,
) -> Tuple[NDArray[np.int32], dict[str, Any]]:
    """Label spatially connected components using the native CCA implementation."""
    coords = data_or_coords.points if hasattr(data_or_coords, "points") else data_or_coords
    coords_array = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError(f"coords must be a (N,3) array, got {coords_array.shape}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and greater than zero")

    labels, stats = _native.connected_components(coords_array, epsilon=float(epsilon))
    labels_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels_array.shape[0] != coords_array.shape[0]:
        raise RuntimeError("Native connected-components output length does not match coords")
    return labels_array, stats
