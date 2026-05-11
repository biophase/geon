from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spatial_subsample_mask(
    coords: NDArray[np.float32],
    min_distance: float,
) -> NDArray[np.bool_]: ...
