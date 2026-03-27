from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class Progress:
    def __init__(self) -> None: ...
    def reset(self, total: int) -> None: ...
    def request_cancel(self) -> None: ...
    def cancelled(self) -> bool: ...
    def done(self) -> int: ...
    def total(self) -> int: ...
    def stage(self) -> str: ...


def segment_superpoints(
    coords: NDArray[np.float32],
    features: NDArray[np.float32] | None = None,
    *,
    k_neighbors: int = ...,
    regularization: float = ...,
    spatial_weight: float = ...,
    cutoff: int = ...,
    iterations: int = ...,
    parallel: bool = ...,
    verbose: bool = ...,
    progress: Progress | None = ...,
) -> tuple[NDArray[np.int32], dict[str, Any]]: ...
