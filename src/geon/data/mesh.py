"""Polygon meshes: geometry only, with no field or segmentation model."""
from pathlib import Path

import numpy as np

from .base import BaseData
from .registry import register_data


@register_data
class MeshData(BaseData):
    def __init__(self, vertices, faces, name="Mesh"):
        super().__init__()
        self.name = str(name)
        self.vertices = np.array(vertices, dtype=np.float64, copy=True)
        if self.vertices.size == 0:
            self.vertices = self.vertices.reshape(0, 3)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Mesh vertices must have shape (N, 3).")
        if not np.isfinite(self.vertices).all():
            raise ValueError("Mesh vertices must be finite.")
        self.faces = []
        for face in faces:
            indices = np.asarray(face)
            if indices.ndim != 1 or len(indices) < 3:
                raise ValueError("Each mesh face must contain at least three vertex indices.")
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Mesh face indices must be integers.")
            if np.any(indices < 0) or np.any(indices >= len(self.vertices)):
                raise ValueError("Mesh face index is outside the vertex array.")
            if len(np.unique(indices)) != len(indices):
                raise ValueError("A mesh face must not repeat vertex indices.")
            self.faces.append(indices.astype(np.int64, copy=True))

    def get_extents(self):
        if not len(self.vertices):
            return None
        low, high = self.vertices.min(axis=0), self.vertices.max(axis=0)
        return tuple(float(v) for pair in zip(low, high) for v in pair)

    def save_hdf5(self, group):
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["name"] = self.name
        group.create_dataset("vertices", data=self.vertices)
        offsets = np.concatenate(([0], np.cumsum([len(f) for f in self.faces], dtype=np.int64)))
        indices = np.concatenate(self.faces) if self.faces else np.empty(0, dtype=np.int64)
        group.create_dataset("face_offsets", data=offsets)
        group.create_dataset("face_indices", data=indices)
        return group

    @classmethod
    def load_hdf5(cls, group):
        offsets = np.asarray(group["face_offsets"])
        indices = np.asarray(group["face_indices"])
        if (offsets.ndim != 1 or not len(offsets) or
                not np.issubdtype(offsets.dtype, np.integer) or
                indices.ndim != 1 or offsets[0] != 0 or
                offsets[-1] != len(indices) or np.any(np.diff(offsets) < 3)):
            raise ValueError("Invalid mesh face offsets.")
        name = group.attrs.get("name", "Mesh")
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        obj = cls(np.asarray(group["vertices"]),
                  [indices[a:b] for a, b in zip(offsets[:-1], offsets[1:])], name)
        obj.id = group.name.rsplit("/", 1)[-1]
        return obj

    @classmethod
    def from_ply(cls, path):
        from geon.io.mesh import ply_to_mesh
        return ply_to_mesh(Path(path))
