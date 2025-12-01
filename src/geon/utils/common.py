import random
import colorsys

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