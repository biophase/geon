import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import h5py
import numpy as np

from .base import BaseData
from .registry import register_data
from .pointcloud import SemanticClass, SemanticSchema
from ..util.common import generate_uuid

CELL_COMPLEX_MAX_DIM=1


@dataclass
class GeometryRef(ABC):
    ref_id: str

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        return {"ID": self.ref_id}
    
@dataclass
class PointCloudRef(GeometryRef):
    field_name: str
    instance_id: int

    def get_properties(self) -> Dict[str, Any]:
        props = super().get_properties()
        props.update({
            "field_name": self.field_name,
            "instance_id": self.instance_id,
        })
        return props


@dataclass(kw_only=True)
class Cell(ABC):
    boundary: List[str] = field(default_factory=list)

    semantic_attributes: Dict[str, int] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    geometry_refs: List[GeometryRef] = field(default_factory=list)
    id: str = field(default_factory=generate_uuid)
    dim: ClassVar[int]
    
    @abstractmethod
    def validate(self, context:"CellComplexData"): 
        assert sum(cell.id == self.id for cell in context.get_cells()) <= 1, \
            "Duplicate ids are not allowed."

        
    
        
    
@dataclass
class VertexCell(Cell):
    dim: ClassVar[int] = 0
    position: Tuple[float, float, float]

    def validate(self, context: "CellComplexData") -> None:
        super().validate(context)
        assert len(self.boundary) == 0, "Vertices are not allowed to have boundaries."
        assert self.dim == 0, "Vertices must have dim=0"

@dataclass(kw_only=True)
class EdgeCell(Cell):
    dim: ClassVar[int] = 1
    boundary: List[str]
    
        
    def validate(self, context: "CellComplexData") -> None:
        super().validate(context)
        assert len(self.boundary) == 2, "Edges must have exactly two boundary cells"
        vert_ids = {cell.id for cell in context.get_cells(dim=0)}
        for b in self.boundary:
            assert b in vert_ids, "Edges must be bound by vertices only"
        assert self.dim == 1, "Edges must have dim=1"
            
    @property
    def end_points(self) -> Tuple[str, str]:
        return self.boundary[0], self.boundary[1]

@register_data
class CellComplexData(BaseData):
    
    def __init__(self, 
                 vertices: Optional[List[VertexCell]] = None,
                 edges: Optional[List[EdgeCell]] = None,
                 semantic_attribute_schemas: Optional[Dict[int, Dict[str, SemanticSchema]]] = None,
                 ):
        super().__init__()
        self.vertices = list(vertices or [])
        self.edges = list(edges or [])
        self.semantic_attribute_schemas: Dict[int, Dict[str, SemanticSchema]] = {
            dim: dict((semantic_attribute_schemas or {}).get(dim, {}))
            for dim in range(CELL_COMPLEX_MAX_DIM + 1)
        }
        self.normalize_and_validate()
        

    def get_cells (self, dim: Optional[int] = None) -> List[Cell]:
        if dim is None:
            return self.vertices + self.edges
        elif dim == 0:
            return self.vertices
        elif dim == 1:
            return self.edges
        else :
            raise ValueError(f"Dim {dim} not supported")

    def get_extents(self) -> tuple[float, float, float, float, float, float] | None:
        if not self.vertices:
            return None
        positions = np.asarray([vertex.position for vertex in self.vertices], dtype=np.float64)
        if positions.size == 0:
            return None
        mins = np.nanmin(positions, axis=0)
        maxs = np.nanmax(positions, axis=0)
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            return None
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    def get_cell_by_id(self, cell_id: str) -> Optional[Cell]:
        return next((cell for cell in self.get_cells() if cell.id == cell_id), None)

    def get_vertex_by_id(self, vertex_id: str) -> Optional[VertexCell]:
        return next((vertex for vertex in self.vertices if vertex.id == vertex_id), None)

    def set_vertex_positions(
        self,
        positions_by_id: Dict[str, Tuple[float, float, float]],
    ) -> None:
        for vertex_id, position in positions_by_id.items():
            vertex = self.get_vertex_by_id(vertex_id)
            if vertex is None:
                continue
            x, y, z = position
            vertex.position = (float(x), float(y), float(z))
        self.normalize_and_validate()
        
    def build_edge(self, id_vert_start:str, id_vert_end:str) -> EdgeCell:
        vert_ids = [cell.id for cell in self.get_cells(dim=0)]
        assert id_vert_start in vert_ids, f"Vertex {id_vert_start} not in Cell Complex"
        assert id_vert_end in vert_ids, f"Vertex {id_vert_end} not in Cell Complex"
        edge = EdgeCell(boundary=[id_vert_start, id_vert_end])
        self.edges.append(edge)
        try:
            self.normalize_and_validate()
        except Exception:
            self.edges.remove(edge)
            raise
        return edge

    def append_edge(self, edge: EdgeCell) -> None:
        self.edges.append(edge)
        try:
            self.normalize_and_validate()
        except Exception:
            self.edges.remove(edge)
            raise

    def remove_edges(self, edge_ids: set[str]) -> List[EdgeCell]:
        removed = [edge for edge in self.edges if edge.id in edge_ids]
        self.edges = [edge for edge in self.edges if edge.id not in edge_ids]
        return removed

    def remove_vertices(self, vertex_ids: set[str]) -> tuple[List[VertexCell], List[EdgeCell]]:
        removed_vertices = [vertex for vertex in self.vertices if vertex.id in vertex_ids]
        self.vertices = [vertex for vertex in self.vertices if vertex.id not in vertex_ids]
        incident_edge_ids = {
            edge.id
            for edge in self.edges
            if any(vertex_id in vertex_ids for vertex_id in edge.boundary)
        }
        removed_edges = self.remove_edges(incident_edge_ids)
        return removed_vertices, removed_edges

    @staticmethod
    def _semantic_class_to_dict(semantic_class: SemanticClass) -> dict[str, Any]:
        return {
            "id": int(semantic_class.id),
            "name": semantic_class.name,
            "color": list(semantic_class.color),
        }

    @staticmethod
    def _semantic_class_from_dict(data: dict[str, Any]) -> SemanticClass:
        color = data.get("color", (204, 204, 204))
        return SemanticClass(
            int(data["id"]),
            str(data["name"]),
            (int(color[0]), int(color[1]), int(color[2])),
        )

    @staticmethod
    def _semantic_attributes_to_dict(attrs: Dict[str, int]) -> dict[str, int]:
        return {str(name): int(value) for name, value in attrs.items()}

    @staticmethod
    def _semantic_attributes_from_dict(data: dict[str, Any]) -> Dict[str, int]:
        attrs: Dict[str, int] = {}
        for name, value in data.items():
            try:
                if isinstance(value, dict) and "value" in value:
                    attrs[str(name)] = int(value["value"]["id"])
                else:
                    attrs[str(name)] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid cell semantic attribute value. Expected integer class ids."
                ) from exc
        return attrs

    @staticmethod
    def _legacy_semantic_attribute_schemas_from_dict(
        data: dict[str, Any],
    ) -> Dict[str, SemanticSchema]:
        schemas: Dict[str, SemanticSchema] = {}
        for name, value in data.items():
            if not isinstance(value, dict) or "schema" not in value:
                continue
            schema = SemanticSchema.from_dict(value.get("schema", {}))
            schema.name = str(value.get("schema_name", schema.name))
            schemas[str(name)] = schema
        return schemas

    @staticmethod
    def _geometry_refs_to_dict(refs: List[GeometryRef]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ref in refs:
            if isinstance(ref, PointCloudRef):
                out.append({
                    "type": "PointCloudRef",
                    "ref_id": ref.ref_id,
                    "field_name": ref.field_name,
                    "instance_id": int(ref.instance_id),
                })
        return out

    @staticmethod
    def _geometry_refs_from_dict(data: list[dict[str, Any]]) -> List[GeometryRef]:
        refs: List[GeometryRef] = []
        for item in data:
            if item.get("type") == "PointCloudRef":
                refs.append(PointCloudRef(
                    ref_id=str(item.get("ref_id", item.get("pointcloud_id", ""))),
                    field_name=str(item["field_name"]),
                    instance_id=int(item["instance_id"]),
                ))
        return refs

    @classmethod
    def _common_cell_kwargs_from_group(cls, group: h5py.Group) -> dict[str, Any]:
        def _decode(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        def _attr_json(name: str, fallback: str) -> Any:
            value = group.attrs.get(name, fallback)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(str(value))

        raw_boundary = group.get("boundary")
        if isinstance(raw_boundary, h5py.Dataset):
            boundary = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in raw_boundary[()]
            ]
        else:
            boundary = []

        return {
            "boundary": boundary,
            "semantic_attributes": cls._semantic_attributes_from_dict(
                _attr_json("semantic_attributes", "{}")
            ),
            "attributes": _attr_json("attributes", "{}"),
            "geometry_refs": cls._geometry_refs_from_dict(_attr_json("geometry_refs", "[]")),
            "id": _decode(group.attrs["id"]),
        }

    @classmethod
    def _legacy_semantic_attribute_schemas_from_group(
        cls,
        group: h5py.Group,
    ) -> Dict[str, SemanticSchema]:
        value = group.attrs.get("semantic_attributes", "{}")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls._legacy_semantic_attribute_schemas_from_dict(json.loads(str(value)))

    @staticmethod
    def _save_common_cell(group: h5py.Group, cell: Cell) -> None:
        group.attrs["id"] = cell.id
        group.attrs["dim"] = cell.dim
        group.attrs["semantic_attributes"] = json.dumps(
            CellComplexData._semantic_attributes_to_dict(cell.semantic_attributes)
        )
        group.attrs["attributes"] = json.dumps(cell.attributes)
        group.attrs["geometry_refs"] = json.dumps(CellComplexData._geometry_refs_to_dict(cell.geometry_refs))
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("boundary", data=np.asarray(cell.boundary, dtype=dt), dtype=dt)

    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id

        schemas_group = group.create_group("semantic_attribute_schemas")
        for dim, schemas_by_name in self.semantic_attribute_schemas.items():
            dim_group = schemas_group.create_group(str(dim))
            for attr_name, schema in schemas_by_name.items():
                attr_group = dim_group.create_group(attr_name)
                schema.save_h5py(attr_group)

        vertices_group = group.create_group("vertices")
        for vertex in self.vertices:
            vertex_group = vertices_group.create_group(vertex.id)
            self._save_common_cell(vertex_group, vertex)
            vertex_group.create_dataset("position", data=np.asarray(vertex.position, dtype=np.float64))

        edges_group = group.create_group("edges")
        for edge in self.edges:
            edge_group = edges_group.create_group(edge.id)
            self._save_common_cell(edge_group, edge)
        return group

    @classmethod
    def load_hdf5(cls, group: h5py.Group) -> "CellComplexData":
        semantic_attribute_schemas: Dict[int, Dict[str, SemanticSchema]] = {
            dim: {} for dim in range(CELL_COMPLEX_MAX_DIM + 1)
        }
        schemas_group = group.get("semantic_attribute_schemas")
        if isinstance(schemas_group, h5py.Group):
            for dim_name, dim_group in schemas_group.items():
                if not isinstance(dim_group, h5py.Group):
                    continue
                dim = int(dim_name)
                semantic_attribute_schemas.setdefault(dim, {})
                for attr_name, attr_group in dim_group.items():
                    if not isinstance(attr_group, h5py.Group):
                        continue
                    semantic_attribute_schemas[dim][str(attr_name)] = (
                        SemanticSchema.from_hdf5_fieldgroup(attr_group)
                    )

        vertices: List[VertexCell] = []
        vertices_group = group.get("vertices")
        if isinstance(vertices_group, h5py.Group):
            for vertex_group in vertices_group.values():
                if not isinstance(vertex_group, h5py.Group):
                    continue
                position_ds = vertex_group.get("position")
                if not isinstance(position_ds, h5py.Dataset):
                    raise ValueError("Vertex cell is missing a 'position' dataset.")
                position_arr = np.asarray(position_ds[()], dtype=np.float64).reshape(3)
                for attr_name, schema in cls._legacy_semantic_attribute_schemas_from_group(vertex_group).items():
                    semantic_attribute_schemas.setdefault(0, {}).setdefault(attr_name, schema)
                vertices.append(VertexCell(
                    position=(float(position_arr[0]), float(position_arr[1]), float(position_arr[2])),
                    **cls._common_cell_kwargs_from_group(vertex_group),
                ))

        edges: List[EdgeCell] = []
        edges_group = group.get("edges")
        if isinstance(edges_group, h5py.Group):
            for edge_group in edges_group.values():
                if not isinstance(edge_group, h5py.Group):
                    continue
                for attr_name, schema in cls._legacy_semantic_attribute_schemas_from_group(edge_group).items():
                    semantic_attribute_schemas.setdefault(1, {}).setdefault(attr_name, schema)
                edges.append(EdgeCell(**cls._common_cell_kwargs_from_group(edge_group)))

        obj = cls(
            vertices=vertices,
            edges=edges,
            semantic_attribute_schemas=semantic_attribute_schemas,
        )
        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = stored_id.decode("utf-8") if isinstance(stored_id, bytes) else str(stored_id)
        obj.normalize_and_validate()
        return obj
    
    @staticmethod
    def _schema_key(schema: SemanticSchema) -> tuple[Any, ...]:
        return (schema.name, schema.signature())

    @staticmethod
    def _default_semantic_value(schema: SemanticSchema) -> SemanticClass:
        classes = sorted(schema.semantic_classes, key=lambda cls: cls.id)
        for cls in classes:
            if cls.id == -1 or cls.name == "_unlabeled":
                return cls
        if not classes:
            raise ValueError(f"Semantic schema '{schema.name}' has no classes.")
        return classes[0]

    def unify_attributes(self) -> None:
        for dim in range(CELL_COMPLEX_MAX_DIM + 1):
            attrs: set[str] = set()
            sem_attrs_by_name = self.semantic_attribute_schemas.setdefault(dim, {})

            for cell in self.get_cells(dim):
                attrs.update(cell.attributes.keys())
                for attr_name, value in list(cell.semantic_attributes.items()):
                    schema = sem_attrs_by_name.get(attr_name)
                    if schema is None:
                        raise ValueError(
                            f"Semantic attribute '{attr_name}' has no schema registered "
                            f"for cell dimension {dim}."
                        )
                    value_ids = {int(sem_cls.id) for sem_cls in schema.semantic_classes}
                    value_id = int(value)
                    if value_id not in value_ids:
                        raise ValueError(
                            f"Semantic attribute '{attr_name}' uses value id {value_id} "
                            f"which is not in schema '{schema.name}'."
                        )
                    cell.semantic_attributes[attr_name] = value_id

            for cell in self.get_cells(dim):
                for attr in attrs:
                    cell.attributes.setdefault(attr, None)
                for sem_attr_name, sem_attr_schema in sem_attrs_by_name.items():
                    cell.semantic_attributes.setdefault(
                        sem_attr_name,
                        int(self._default_semantic_value(sem_attr_schema).id),
                    )

    def normalize_and_validate(self) -> None:
        self.unify_attributes()
        for cell in self.get_cells():
            cell.validate(self)

    def iter_semantic_attributes(self):
        for dim in range(CELL_COMPLEX_MAX_DIM + 1):
            for attr_name, schema in self.semantic_attribute_schemas.get(dim, {}).items():
                yield dim, attr_name, schema

    def get_matching_semantic_attribute_schemas(
        self,
        schema: SemanticSchema,
    ) -> dict[str, SemanticSchema]:
        matches: dict[str, SemanticSchema] = {}
        for dim, attr_name, candidate in self.iter_semantic_attributes():
            if candidate.name != schema.name:
                continue
            if candidate.signature() != schema.signature():
                continue
            key = f"{dim}/{attr_name}/{candidate.name}"
            matches[key] = candidate
        return matches

    def remap_semantic_attribute(
        self,
        dim: int,
        attr_name: str,
        old_to_new_ids: list[tuple[int, int]],
        new_schema: SemanticSchema,
    ) -> None:
        mapping = {int(old_id): int(new_id) for old_id, new_id in old_to_new_ids}
        default_value = self._default_semantic_value(new_schema)
        new_ids = {int(sem_cls.id) for sem_cls in new_schema.semantic_classes}

        for cell in self.get_cells(dim):
            if attr_name not in cell.semantic_attributes:
                continue
            target_id = mapping.get(int(cell.semantic_attributes[attr_name]), int(default_value.id))
            if target_id not in new_ids:
                target_id = int(default_value.id)
            cell.semantic_attributes[attr_name] = int(target_id)
        self.semantic_attribute_schemas.setdefault(dim, {})[attr_name] = new_schema
        self.normalize_and_validate()

    def add_semantic_attribute(
        self,
        dim: int,
        attr_name: str,
        schema: SemanticSchema,
    ) -> None:
        attr_name = attr_name.strip()
        if not attr_name:
            raise ValueError("Semantic attribute name cannot be empty.")
        if any(ch.isspace() for ch in attr_name):
            raise ValueError("Semantic attribute name cannot contain spaces.")
        schemas = self.semantic_attribute_schemas.setdefault(dim, {})
        if attr_name in schemas:
            raise ValueError(
                f"Semantic attribute '{attr_name}' already exists for cell dimension {dim}."
            )
        schemas[attr_name] = schema
        default_value = int(self._default_semantic_value(schema).id)
        for cell in self.get_cells(dim):
            cell.semantic_attributes[attr_name] = default_value
        self.normalize_and_validate()

    def delete_semantic_attribute(self, dim: int, attr_name: str) -> None:
        schemas = self.semantic_attribute_schemas.setdefault(dim, {})
        if attr_name not in schemas:
            raise ValueError(
                f"Semantic attribute '{attr_name}' does not exist for cell dimension {dim}."
            )
        schemas.pop(attr_name)
        for cell in self.get_cells(dim):
            cell.semantic_attributes.pop(attr_name, None)
        self.normalize_and_validate()

            
    @classmethod
    def from_txt(cls, path: str, delimiter: str = ",") -> "CellComplexData":
        data = np.atleast_2d(np.loadtxt(path, delimiter=delimiter))
        if data.shape[1] != 3:
            raise ValueError(f"Expected 3 coordinate columns, got {data.shape[1]}.")
        return cls(
            vertices=[
                VertexCell(position=tuple(float(x) for x in pos))
                for pos in data
            ]
        )


CellComplex = CellComplexData
