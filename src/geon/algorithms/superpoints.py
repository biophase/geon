from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from geon._native import superpoints as _native

if TYPE_CHECKING:
    from ..data.pointcloud import PointCloudData

Progress = _native.Progress


def _as_coords(
    data_or_coords: "PointCloudData" | NDArray[np.float32],
) -> NDArray[np.float32]:
    coords = data_or_coords.points if hasattr(data_or_coords, "points") else data_or_coords
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be a (N,3) array, got {arr.shape}")
    return arr


def _supported_field(field_type: object) -> bool:
    return getattr(field_type, "name", None) in {
        "SCALAR",
        "VECTOR",
        "NORMAL",
        "INTENSITY",
    }


def _build_feature_matrix_from_fields(
    data: "PointCloudData",
    field_names: Iterable[str],
) -> NDArray[np.float32] | None:
    parts: list[NDArray[np.float32]] = []
    for field_name in field_names:
        fields = data.get_fields(names=field_name)
        if not fields:
            raise ValueError(f"feature field '{field_name}' was not found")
        field = fields[0]
        if not _supported_field(field.field_type):
            raise ValueError(
                f"field '{field_name}' has unsupported type {field.field_type.name}"
            )
        arr = np.asarray(field.data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != data.points.shape[0]:
            raise ValueError(
                f"field '{field_name}' must have shape (N,F), got {arr.shape}"
            )
        parts.append(arr)
    if not parts:
        return None
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def segment_superpoints(
    data_or_coords: "PointCloudData" | NDArray[np.float32],
    *,
    feature_field_names: Optional[Iterable[str]] = None,
    features: Optional[NDArray[np.float32]] = None,
    k_neighbors: int = 10,
    regularization: float = 0.05,
    spatial_weight: float = 1.0,
    cutoff: int = 10,
    iterations: int = 10,
    parallel: bool = True,
    verbose: bool = False,
    progress: Optional[Progress] = None,
) -> Tuple[NDArray[np.int32], dict[str, Any]]:
    coords = _as_coords(data_or_coords)
    if feature_field_names is not None and features is not None:
        raise ValueError("Pass either feature_field_names or features, not both")

    features_arr: NDArray[np.float32] | None = None
    if feature_field_names is not None:
        if not hasattr(data_or_coords, "get_fields") or not hasattr(data_or_coords, "points"):
            raise ValueError("feature_field_names requires a PointCloudData input")
        features_arr = _build_feature_matrix_from_fields(data_or_coords, feature_field_names)
    elif features is not None:
        features_arr = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
        if features_arr.ndim != 2 or features_arr.shape[0] != coords.shape[0]:
            raise ValueError(
                f"features must have shape (N,F) with N={coords.shape[0]}, got {features_arr.shape}"
            )

    labels, stats = _native.segment_superpoints(
        coords,
        features_arr,
        k_neighbors=int(k_neighbors),
        regularization=float(regularization),
        spatial_weight=float(spatial_weight),
        cutoff=int(cutoff),
        iterations=int(iterations),
        parallel=bool(parallel),
        verbose=bool(verbose),
        progress=progress,
    )
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    return labels, stats
