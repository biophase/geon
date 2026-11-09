"""Polygon mesh topology and shared point-cloud field types at four levels."""
from pathlib import Path
from enum import Enum
import json
import h5py

import numpy as np

from .base import BaseData
from .registry import register_data
from .pointcloud import PointCloudData, FieldType
from geon.util.common import decode_utf8


class MeshLevel(str, Enum):
    VERTEX = "vertex"
    EDGE = "edge"
    FACE = "face"
    OBJECT = "object"


@register_data
class MeshData(BaseData):
    def __init__(self, vertices, faces, name="Mesh", *, face_object_ids=None):
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

        pairs = [np.sort(np.column_stack((face, np.roll(face, -1))), axis=1) for face in self.faces]
        self.edges = (np.unique(np.concatenate(pairs), axis=0) if pairs
                      else np.empty((0, 2), dtype=np.int64))
        if face_object_ids is None:
            face_object_ids = np.zeros(len(self.faces), dtype=np.int64)
        object_array = np.asarray(face_object_ids)
        if object_array.shape != (len(self.faces),) or not np.issubdtype(object_array.dtype, np.integer):
            raise ValueError("face_object_ids must contain one integer per face.")
        self.face_object_ids = object_array.astype(np.int64, copy=True)
        self.object_ids = np.unique(self.face_object_ids)
        # These containers reuse the existing field creation, schema and metadata code.
        # Zero coordinate columns allocate no extra geometry.
        self._field_sets = {level: PointCloudData(np.empty((self.element_count(level), 0)))
                            for level in MeshLevel}
        self.metadata = {}

    def element_count(self, level):
        return {MeshLevel.VERTEX: len(self.vertices), MeshLevel.EDGE: len(self.edges),
                MeshLevel.FACE: len(self.faces), MeshLevel.OBJECT: len(self.object_ids)}[MeshLevel(level)]

    def field_container(self, level=MeshLevel.FACE):
        return self._field_sets[MeshLevel(level)]

    def add_field(self, name=None, data=None, field_type=None, *, level=MeshLevel.FACE, **kwargs):
        level = MeshLevel(level)
        container = self.field_container(level)
        if name is not None and (not str(name).strip() or "/" in str(name)):
            raise ValueError("Field names must be nonempty and must not contain '/'.")
        if name is None:
            index = 0
            while f"Field_{index:04}" in container.field_names:
                index += 1
            name = f"Field_{index:04}"
        if name in container.field_names:
            raise ValueError(f"Duplicate {level.value} field: {name}")
        if data is not None:
            data = np.asarray(data)
            if data.ndim == 0 or len(data) != self.element_count(level):
                raise ValueError(f"Field length must match the {level.value} count.")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        container.add_field(name=name, data=data, field_type=field_type, **kwargs)
        return container.get_fields()[-1]

    def get_fields(self, names=None, field_type=None, field_index=None, *, level=None):
        levels = list(MeshLevel) if level is None else [MeshLevel(level)]
        fields = [field for item in levels for field in self.field_container(item).get_fields(
            names=names, field_type=field_type)]
        return fields if field_index is None else [fields[field_index]]

    def remove_fields(self, names=None, field_type=None, *, level=MeshLevel.FACE):
        self.field_container(level).remove_fields(names=names, field_type=field_type)

    def field_level(self, field):
        for level in MeshLevel:
            if any(candidate is field for candidate in self.field_container(level).get_fields()):
                return level
        raise ValueError("Field does not belong to this mesh.")

    @property
    def field_names(self):
        return [field.name for field in self.get_fields()]

    def face_values(self, field):
        """Project a field to faces, voting on incident vertices or edges if needed."""
        level = self.field_level(field)
        columns = field.data.shape[1] if field.data.ndim > 1 else 1
        values = field.data.reshape(len(field.data), columns)
        if level == MeshLevel.FACE:
            return values.copy()
        if level == MeshLevel.OBJECT:
            return values[np.searchsorted(self.object_ids, self.face_object_ids)]
        edge_lookup = {tuple(edge): i for i, edge in enumerate(self.edges)} if level == MeshLevel.EDGE else None
        output = []
        for face in self.faces:
            indices = face if level == MeshLevel.VERTEX else [
                edge_lookup[tuple(sorted((int(a), int(b))))] for a, b in zip(face, np.roll(face, -1))]
            unique, counts = np.unique(values[indices], axis=0, return_counts=True)
            output.append(unique[np.argmax(counts)])
        return np.asarray(output, dtype=values.dtype).reshape(len(self.faces), values.shape[1])

    def extract_selection(self, selection, *, level=MeshLevel.FACE):
        from copy import deepcopy
        level = MeshLevel(level)
        selected = np.asarray(selection, dtype=np.int64)
        if level == MeshLevel.FACE:
            faces = np.unique(selected)
        elif level == MeshLevel.OBJECT:
            faces = np.flatnonzero(np.isin(self.face_object_ids, self.object_ids[selected]))
        elif level == MeshLevel.VERTEX:
            faces = np.array([i for i, face in enumerate(self.faces) if np.isin(face, selected).all()], dtype=np.int64)
        else:
            edges = {tuple(edge) for edge in self.edges[selected]}
            faces = np.array([i for i, face in enumerate(self.faces) if all(
                tuple(sorted((int(a), int(b)))) in edges for a, b in zip(face, np.roll(face, -1)))], dtype=np.int64)
        if not len(faces):
            raise ValueError("Select at least one complete face to extract a mesh.")
        vertices = np.unique(np.concatenate([self.faces[i] for i in faces]))
        result = MeshData(self.vertices[vertices], [np.searchsorted(vertices, self.faces[i]) for i in faces],
                          self.name+" (selection)", face_object_ids=self.face_object_ids[faces])
        edge_lookup = {tuple(edge): i for i, edge in enumerate(self.edges)}
        edge_rows = [edge_lookup[tuple(vertices[edge])] for edge in result.edges]
        indices = {MeshLevel.VERTEX: vertices, MeshLevel.EDGE: edge_rows, MeshLevel.FACE: faces,
                   MeshLevel.OBJECT: np.searchsorted(self.object_ids, result.object_ids)}
        for field_level in MeshLevel:
            for source in self.get_fields(level=field_level):
                field = deepcopy(source)
                field.data = source.data[indices[field_level]].copy()
                result.field_container(field_level)._fields.append(field)
        result.metadata = deepcopy(self.metadata)
        return result

    def get_extents(self):
        if not len(self.vertices):
            return None
        low, high = self.vertices.min(axis=0), self.vertices.max(axis=0)
        return tuple(float(v) for pair in zip(low, high) for v in pair)

    def save_hdf5(self, group):
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["name"] = self.name
        group.create_dataset("vertices", data=self.vertices)
        group.create_dataset("face_object_ids", data=self.face_object_ids)
        group.create_dataset("metadata", data=json.dumps(self.metadata), dtype=h5py.string_dtype("utf-8"))
        fields_group = group.create_group("fields")
        for level in MeshLevel:
            subgroup = fields_group.create_group(level.value, track_order=True)
            for field in self.get_fields(level=level):
                field.save_hdf5(subgroup)
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
                  [indices[a:b] for a, b in zip(offsets[:-1], offsets[1:])], name,
                  face_object_ids=np.asarray(group["face_object_ids"]) if "face_object_ids" in group else None)
        metadata = group["metadata"][()] if "metadata" in group else group.attrs.get("metadata", "{}")
        obj.metadata = json.loads(decode_utf8(metadata))
        fields = group.get("fields")
        if isinstance(fields, h5py.Group):
            for level in MeshLevel:
                subgroup = fields.get(level.value)
                if not isinstance(subgroup, h5py.Group):
                    continue
                for field_group in subgroup.values():
                    field_type = FieldType[decode_utf8(field_group.attrs.get("field_type", "SCALAR"))]
                    field = FieldType.get_class(field_type).from_hdf5_fieldgroup(field_group)
                    if len(field.data) != obj.element_count(level):
                        raise ValueError(f"Invalid {level.value} field length: {field.name}")
                    obj.field_container(level)._fields.append(field)
        obj.id = group.name.rsplit("/", 1)[-1]
        return obj

    @classmethod
    def from_ply(cls, path):
        from geon.io.mesh import ply_to_mesh
        return ply_to_mesh(Path(path))
