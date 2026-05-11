from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geon._native import subsampling as _native


def spatial_subsample_mask(
    coords: NDArray[np.float32],
    min_distance: float,
) -> NDArray[np.bool_]:
    coords_f32 = np.asarray(coords, dtype=np.float32)
    return _native.spatial_subsample_mask(coords_f32, float(min_distance))
