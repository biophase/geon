import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import h5py
import numpy as np

from .base import BaseData
from .registry import register_data
from .pointcloud import SemanticClass
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
    semantic_type: Optional[SemanticClass] = None
    attributes: dict[str, Any] = field(default_factory=dict)
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
                 edges: Optional[List[EdgeCell]] = None
                 ):
        super().__init__()
        self.vertices = list(vertices or [])
        self.edges = list(edges or [])
        

    def get_cells (self, dim: Optional[int] = None) -> List[Cell]:
        if dim is None:
            return self.vertices + self.edges
        elif dim == 0:
            return self.vertices
        elif dim == 1:
            return self.edges
        else :
            raise ValueError(f"Dim {dim} not supported")

    def get_cell_by_id(self, cell_id: str) -> Optional[Cell]:
        return next((cell for cell in self.get_cells() if cell.id == cell_id), None)
        
    def build_edge(self, id_vert_start:str, id_vert_end:str) -> EdgeCell:
        vert_ids = [cell.id for cell in self.get_cells(dim=0)]
        assert id_vert_start in vert_ids, f"Vertex {id_vert_start} not in Cell Complex"
        assert id_vert_end in vert_ids, f"Vertex {id_vert_end} not in Cell Complex"
        edge = EdgeCell(boundary=[id_vert_start, id_vert_end])
        edge.validate(self)
        self.edges.append(edge)
        return edge

    def append_edge(self, edge: EdgeCell) -> None:
        edge.validate(self)
        self.edges.append(edge)

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
    def _semantic_to_dict(semantic_type: Optional[SemanticClass]) -> Optional[dict[str, Any]]:
        if semantic_type is None:
            return None
        return {
            "id": int(semantic_type.id),
            "name": semantic_type.name,
            "color": list(semantic_type.color),
        }

    @staticmethod
    def _semantic_from_dict(data: Optional[dict[str, Any]]) -> Optional[SemanticClass]:
        if data is None:
            return None
        color = data.get("color", (204, 204, 204))
        return SemanticClass(
            int(data["id"]),
            str(data["name"]),
            (int(color[0]), int(color[1]), int(color[2])),
        )

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
            "semantic_type": cls._semantic_from_dict(_attr_json("semantic_type", "null")),
            "attributes": _attr_json("attributes", "{}"),
            "geometry_refs": cls._geometry_refs_from_dict(_attr_json("geometry_refs", "[]")),
            "id": _decode(group.attrs["id"]),
        }

    @staticmethod
    def _save_common_cell(group: h5py.Group, cell: Cell) -> None:
        group.attrs["id"] = cell.id
        group.attrs["dim"] = cell.dim
        group.attrs["semantic_type"] = json.dumps(CellComplexData._semantic_to_dict(cell.semantic_type))
        group.attrs["attributes"] = json.dumps(cell.attributes)
        group.attrs["geometry_refs"] = json.dumps(CellComplexData._geometry_refs_to_dict(cell.geometry_refs))
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("boundary", data=np.asarray(cell.boundary, dtype=dt), dtype=dt)

    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id

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
                edges.append(EdgeCell(**cls._common_cell_kwargs_from_group(edge_group)))

        obj = cls(vertices=vertices, edges=edges)
        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = stored_id.decode("utf-8") if isinstance(stored_id, bytes) else str(stored_id)
        return obj
    
    
    def unify_attributes(self) -> None:
        for dim in range(CELL_COMPLEX_MAX_DIM + 1):
            attrs: set[str] = set()
            for cell in self.get_cells(dim):
                attrs.update(cell.attributes.keys())
            for cell in self.get_cells(dim):
                for attr in attrs:
                    cell.attributes.setdefault(attr, None)
            
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
