import vtk
from geon.data.definitions import ColorMap
import numpy as np
from dataclasses import dataclass

from typing import Tuple
 

def build_vtk_color_transfer_function(
    colormap: ColorMap,
) -> tuple[vtk.vtkColorTransferFunction, Tuple[float, float]]:
    """
    Build a vtkColorTransferFunction from a ColorMap and return it
    together with the scalar range (min, max) for convenience.
    """

    positions = np.asarray(colormap.color_positions, dtype=float)
    colors = np.asarray(colormap.color_definitions, dtype=float)

    if positions.ndim != 1:
        raise ValueError("color_positions must be a 1D array of scalars.")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must be a 2D array of shape (N, 3).")
    if positions.shape[0] != colors.shape[0]:
        raise ValueError(
            f"color_positions and colors must have the same length, "
            f"got {positions.shape[0]} and {colors.shape[0]}."
        )

    # Sort by position to ensure monotonicity
    sort_idx = np.argsort(positions)
    positions = positions[sort_idx]
    colors = colors[sort_idx]

    # Create the color transfer function
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetAllowDuplicateScalars(False)   # keeps things clean

    if colormap.color_type == 'rgb':
        ctf.SetColorSpaceToRGB()
        for x, (r, g, b) in zip(positions, colors):
            ctf.AddRGBPoint(float(x), float(r), float(g), float(b))

    elif colormap.color_type == 'hsv':
        ctf.SetColorSpaceToHSV()
        for x, (h, s, v) in zip(positions, colors):
            ctf.AddHSVPoint(float(x), float(h), float(s), float(v))

    else:
        raise ValueError(f"Unsupported color_type: {colormap.color_type}")

    scalar_range = (float(positions[0]), float(positions[-1]))
    return ctf, scalar_range

