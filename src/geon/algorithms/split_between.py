from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from geon._native import split_between as _native

Progress = _native.Progress
SplitCondition = Literal["plane_distance", "nearest_neighbor"]


def split_between(
    coords: NDArray[np.float32],
    working_indices: NDArray[np.integer],
    a_indices: NDArray[np.integer],
    b_indices: NDArray[np.integer],
    condition: SplitCondition,
    progress: Progress | None = None,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    points = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coords must have shape (N,3), got {points.shape}")

    def indices(value: NDArray[np.integer], name: str) -> NDArray[np.int64]:
        result = np.asarray(value)
        if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        return np.ascontiguousarray(result, dtype=np.int64)

    assignment, stats = _native.split_between(
        points,
        indices(working_indices, "working_indices"),
        indices(a_indices, "a_indices"),
        indices(b_indices, "b_indices"),
        condition,
        progress,
    )
    return np.asarray(assignment, dtype=bool).reshape(-1), stats
