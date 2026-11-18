from typing import Callable
import numpy as np
from numpy.typing import NDArray

def edges(points: NDArray[np.float64], labels: NDArray[np.int64], radius: float,
          progress: Callable[[str, int, int], None]) -> NDArray[np.uint64]:
    """Sorted unique packed superpoint edges from exact native radius searches."""
    ...
