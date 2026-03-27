from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


class Progress:
    def reset(self, total: int) -> None: ...
    def request_cancel(self) -> None: ...
    def cancelled(self) -> bool: ...
    def done(self) -> int: ...
    def total(self) -> int: ...
    def chunk_statuses(self) -> list[dict[str, Any]]: ...


class SeededGrower:
    def __init__(
        self,
        coords: NDArray[np.float32],
        normals: NDArray[np.float32] | None = ...,
        *,
        normal_mode: str = ...,
        params: dict[str, Any] = ...,
    ) -> None: ...
    def grow(self, seed_index: int) -> Tuple[NDArray[np.int32], dict[str, Any]]: ...


def estimate_parameters(
    coords: NDArray[np.float32],
    *,
    sample_size: int = ...,
    seed: int = ...,
) -> Tuple[float, int, float, dict[str, Any]]: ...


def segment_planar_regions(
    coords: NDArray[np.float32],
    normals: NDArray[np.float32] | None = ...,
    *,
    normal_mode: str = ...,
    params: dict[str, Any] = ...,
    chunking: dict[str, Any] = ...,
    merge: dict[str, Any] = ...,
    progress: Progress | None = ...,
) -> Tuple[NDArray[np.int32], dict[str, Any]]: ...
