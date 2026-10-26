"""Geometry-only PLY import; triangles, quads and mixed polygons are preserved."""
from pathlib import Path

import numpy as np
from plyfile import PlyData

from geon.data.mesh import MeshData


def ply_to_mesh(path):
    path = Path(path)
    ply = PlyData.read(str(path))
    if "vertex" not in ply or "face" not in ply:
        raise ValueError("A mesh PLY must contain vertex and face elements.")
    vertices = ply["vertex"].data
    if not all(axis in vertices.dtype.names for axis in ("x", "y", "z")):
        raise ValueError("PLY vertices must contain x, y and z coordinates.")
    faces = ply["face"].data
    index_name = next((key for key in ("vertex_indices", "vertex_index")
                       if key in faces.dtype.names), None)
    if index_name is None:
        raise ValueError("PLY faces must contain a vertex_indices list.")
    if not len(faces):
        raise ValueError("The PLY contains no mesh faces.")
    return MeshData(np.column_stack([vertices[axis] for axis in ("x", "y", "z")]),
                    faces[index_name], path.stem)
