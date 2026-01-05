import random
import colorsys
import numpy as np
from numpy.typing import NDArray

import geon.core

def decode_utf8(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def generate_vibrant_color():
    """Generate a random vibrant color (avoid grays, blacks, whites)."""
    def _to_int(x:float):
        return int(x*255)
    h = random.random()
    s = 0.9
    v = 0.9
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r = _to_int(r)
    g = _to_int(g)
    b = _to_int(b)
    return (r, g, b)

def blend_colors(c1, c2, t):
    """Linearly blend two RGB colors with blend factor t (0<=t<=1)."""
    return tuple((1 - t) * a + t * b for a, b in zip(c1, c2))

def bool_op_index_mask(inds_a: NDArray[np.int32], 
                       inds_b: NDArray[np.int32], 
                       bool_op: geon.core.Boolean)-> NDArray[np.int32]:
    a = np.asarray(inds_a, dtype=np.int32)
    b = np.asarray(inds_b, dtype=np.int32)
    if bool_op == geon.core.Boolean.UNION:
        out = np.union1d(a, b)
    elif bool_op == geon.core.Boolean.DIFFERENCE:
        out = np.setdiff1d(a, b)
    elif bool_op == geon.core.Boolean.INTERSECTION:
        out = np.intersect1d(a, b)
    elif bool_op == geon.core.Boolean.EXCLUSION:
        out = np.setxor1d(a, b)
    elif bool_op == geon.core.Boolean.OVERWRITE:
        out = b
    else:
        raise ValueError(f"Unsupported boolean op: {bool_op}")
    return out.astype(np.int32, copy=False)
