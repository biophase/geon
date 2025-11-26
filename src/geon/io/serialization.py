import base64
import numpy as np

def ndarray_to_base64(arr: np.ndarray) -> dict:
    """
    Convert a numpy array to a base64 representation suitable for JSON.
    """
    return {
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "data": base64.b64encode(arr.tobytes()).decode("ascii")
    }