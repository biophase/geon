from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

from geon.util.common import decode_utf8, generate_uuid

from .base import BaseData
from .pointcloud import SemanticSchema
from .registry import register_data


FACE_NAMES = ("bottom", "top", "xmin", "xmax", "ymin", "ymax")




@dataclass
class BoundingBox:
    id: str = field(default_factory=generate_uuid)
    center_bottom_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    width: float = 1.0
    depth: float = 1.0
    height: float = 1.0
    semantic_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _rotation_matrix(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
        ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64)
        rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64)
        return rz @ ry @ rx

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        return self._rotation_matrix(self.yaw, self.pitch, self.roll)

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (float(self.width), float(self.depth), float(self.height))

    def transform_matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = np.asarray(self.center_bottom_xyz, dtype=np.float64)
        return matrix

    def local_corners(self) -> NDArray[np.float64]:
        hw = float(self.width) * 0.5
        hd = float(self.depth) * 0.5
        h = float(self.height)
        return np.asarray(
            (
                (-hw, -hd, 0.0),
                (hw, -hd, 0.0),
                (hw, hd, 0.0),
                (-hw, hd, 0.0),
                (-hw, -hd, h),
                (hw, -hd, h),
                (hw, hd, h),
                (-hw, hd, h),
            ),
            dtype=np.float64,
        )

    def corners(self) -> NDArray[np.float64]:
        origin = np.asarray(self.center_bottom_xyz, dtype=np.float64)
        return origin[None, :] + self.local_corners() @ self.rotation_matrix.T

    def axes(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r = self.rotation_matrix
        return r[:, 0], r[:, 1], r[:, 2]

    def face_quads(self) -> Dict[str, Tuple[int, int, int, int]]:
        return {
            "bottom": (0, 1, 2, 3),
            "top": (4, 5, 6, 7),
            "xmin": (0, 3, 7, 4),
            "xmax": (1, 2, 6, 5),
            "ymin": (0, 1, 5, 4),
            "ymax": (3, 2, 6, 7),
        }

    def face_center(self, face: str) -> NDArray[np.float64]:
        corners = self.corners()
        return corners[np.asarray(self.face_quads()[face], dtype=np.int64)].mean(axis=0)

    def face_normal(self, face: str) -> NDArray[np.float64]:
        axis_x, axis_y, axis_z = self.axes()
        normals = {
            "bottom": -axis_z,
            "top": axis_z,
            "xmin": -axis_x,
            "xmax": axis_x,
            "ymin": -axis_y,
            "ymax": axis_y,
        }
        normal = np.asarray(normals[face], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        return normal / norm if norm > 1e-12 else normal

    def adjust_face(self, face: str, delta: float, min_size: float = 1e-6) -> None:
        if face not in FACE_NAMES:
            raise ValueError(f"Unknown box face '{face}'.")
        delta = float(delta)
        axis_x, axis_y, axis_z = self.axes()
        center = np.asarray(self.center_bottom_xyz, dtype=np.float64)
        if face == "top":
            new_height = max(min_size, float(self.height) + delta)
            applied = new_height - float(self.height)
            self.height = new_height
            return
        if face == "bottom":
            new_height = max(min_size, float(self.height) + delta)
            applied = new_height - float(self.height)
            self.height = new_height
            center -= axis_z * applied
        elif face == "xmax":
            new_width = max(min_size, float(self.width) + delta)
            applied = new_width - float(self.width)
            self.width = new_width
            center += axis_x * (applied * 0.5)
        elif face == "xmin":
            new_width = max(min_size, float(self.width) + delta)
            applied = new_width - float(self.width)
            self.width = new_width
            center -= axis_x * (applied * 0.5)
        elif face == "ymax":
            new_depth = max(min_size, float(self.depth) + delta)
            applied = new_depth - float(self.depth)
            self.depth = new_depth
            center += axis_y * (applied * 0.5)
        elif face == "ymin":
            new_depth = max(min_size, float(self.depth) + delta)
            applied = new_depth - float(self.depth)
            self.depth = new_depth
            center -= axis_y * (applied * 0.5)
        self.center_bottom_xyz = tuple(float(v) for v in center)

    def normalize_points_flat(self, points:np.ndarray) -> np.ndarray:
        """
        Assumes BBox is flat (pitch, roll = 0)
        """
        rotAndScale = np.array([
            [np.cos(self.yaw)*(1/self.width),   np.cos(self.yaw + np.pi/2)*(1/self.depth),  0],
            [np.sin(self.yaw)*(1/self.width),   np.sin(self.yaw + np.pi/2)*(1/self.depth),  0],
            [0,                                 0,                                          1/ self.height]
        ])
        if points.shape[1] != 3:
            raise ValueError(f"Expected points shape N,3 but got {points.shape}")

        points_centered = points - self.center_bottom_xyz
        return points_centered @ rotAndScale


    @classmethod
    def from_horizontal_corners(
        cls,
        c1: Tuple[float, float, float],
        c2: Tuple[float, float, float],
        c3: Tuple[float, float, float],
        bottom_z: float,
        top_z: float,
        *,
        box_id: Optional[str] = None,
        semantic_id: Optional[int] = None,
    ) -> "BoundingBox":
        p1 = np.asarray((c1[0], c1[1]), dtype=np.float64)
        p2 = np.asarray((c2[0], c2[1]), dtype=np.float64)
        p3 = np.asarray((c3[0], c3[1]), dtype=np.float64)
        v1 = p2 - p1
        width = float(np.linalg.norm(v1))
        if width <= 1e-12:
            raise ValueError("The first two bounding-box corners must not coincide.")
        axis_x = v1 / width
        axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float64)
        depth_signed = float(np.dot(p3 - p1, axis_y))
        if abs(depth_signed) <= 1e-12:
            raise ValueError("The third bounding-box corner must define a non-zero depth.")
        center_xy = p1 + 0.5 * v1 + 0.5 * depth_signed * axis_y
        yaw = math.atan2(float(axis_x[1]), float(axis_x[0]))
        bottom = float(min(bottom_z, top_z))
        top = float(max(bottom_z, top_z))
        return cls(
            id=box_id or generate_uuid(),
            center_bottom_xyz=(float(center_xy[0]), float(center_xy[1]), bottom),
            yaw=yaw,
            pitch=0.0,
            roll=0.0,
            width=width,
            depth=abs(depth_signed),
            height=max(top - bottom, 1e-6),
            semantic_id=semantic_id,
        )


@register_data
class BoundingBoxData(BaseData):
    type_id = "BoundingBoxData"

    def __init__(
        self,
        boxes: Optional[List[BoundingBox]] = None,
        schema: Optional[SemanticSchema] = None,
    ):
        super().__init__()
        self.boxes: List[BoundingBox] = list(boxes or [])
        self.schema = schema

    @property
    def box_count(self) -> int:
        return len(self.boxes)

    def get_extents(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        if not self.boxes:
            return None
        corners = np.concatenate([box.corners() for box in self.boxes], axis=0)
        mins = np.nanmin(corners, axis=0)
        maxs = np.nanmax(corners, axis=0)
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

    def get_box(self, box_id: str) -> Optional[BoundingBox]:
        return next((box for box in self.boxes if box.id == box_id), None)

    def append_box(self, box: BoundingBox) -> None:
        if self.get_box(box.id) is not None:
            raise ValueError(f"Duplicate bounding box id '{box.id}'.")
        self.boxes.append(box)

    def remove_box(self, box_id: str) -> Optional[BoundingBox]:
        box = self.get_box(box_id)
        if box is None:
            return None
        self.boxes = [candidate for candidate in self.boxes if candidate.id != box_id]
        return box

    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id
        dt = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("box_ids", data=[b.id for b in self.boxes], dtype=dt)
        group.create_dataset("center_bottom_xyz", data=np.asarray([b.center_bottom_xyz for b in self.boxes], dtype=np.float64).reshape((-1, 3)))
        group.create_dataset("rotations_yaw_pitch_roll", data=np.asarray([(b.yaw, b.pitch, b.roll) for b in self.boxes], dtype=np.float64).reshape((-1, 3)))
        group.create_dataset("dimensions_width_depth_height", data=np.asarray([b.dimensions for b in self.boxes], dtype=np.float64).reshape((-1, 3)))
        sem = np.asarray([b.semantic_id if b.semantic_id is not None else -1 for b in self.boxes], dtype=np.int32)
        group.create_dataset("semantic_id", data=sem)
        group.create_dataset("attributes_json", data=[json.dumps(b.attributes) for b in self.boxes], dtype=dt)
        if self.schema is not None:
            schema_group = group.create_group("schema")
            self.schema.save_h5py(schema_group)
        return group

    @classmethod
    def load_hdf5(cls, group: h5py.Group) -> "BoundingBoxData":
        ids_ds = group.get("box_ids")
        centers_ds = group.get("center_bottom_xyz")
        rotations_ds = group.get("rotations_yaw_pitch_roll")
        dims_ds = group.get("dimensions_width_depth_height")
        if not all(isinstance(ds, h5py.Dataset) for ds in (ids_ds, centers_ds, rotations_ds, dims_ds)):
            raise ValueError("BoundingBoxData HDF5 group is missing required datasets.")
        ids = [decode_utf8(v) for v in ids_ds[()]]  # type: ignore[union-attr]
        centers = np.asarray(centers_ds[()], dtype=np.float64)  # type: ignore[union-attr]
        rotations = np.asarray(rotations_ds[()], dtype=np.float64)  # type: ignore[union-attr]
        dims = np.asarray(dims_ds[()], dtype=np.float64)  # type: ignore[union-attr]
        sem_ds = group.get("semantic_id")
        sem_ids = np.asarray(sem_ds[()], dtype=np.int32) if isinstance(sem_ds, h5py.Dataset) else np.full((len(ids),), -1, dtype=np.int32)
        attr_ds = group.get("attributes_json")
        attr_values = [decode_utf8(v) for v in attr_ds[()]] if isinstance(attr_ds, h5py.Dataset) else ["{}"] * len(ids)
        boxes: List[BoundingBox] = []
        for i, box_id in enumerate(ids):
            try:
                attrs = json.loads(attr_values[i])
            except json.JSONDecodeError:
                attrs = {}
            boxes.append(
                BoundingBox(
                    id=box_id,
                    center_bottom_xyz=tuple(float(v) for v in centers[i]),
                    yaw=float(rotations[i, 0]),
                    pitch=float(rotations[i, 1]),
                    roll=float(rotations[i, 2]),
                    width=float(dims[i, 0]),
                    depth=float(dims[i, 1]),
                    height=float(dims[i, 2]),
                    semantic_id=None if int(sem_ids[i]) < 0 else int(sem_ids[i]),
                    attributes=attrs if isinstance(attrs, dict) else {},
                )
            )
        schema = None
        schema_group = group.get("schema")
        if isinstance(schema_group, h5py.Group):
            schema = SemanticSchema.from_hdf5_fieldgroup(schema_group)
        obj = cls(boxes=boxes, schema=schema)
        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = decode_utf8(stored_id)
        return obj
