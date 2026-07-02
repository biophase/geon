from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import vtk

from .base import BaseData
from .registry import register_data


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _float_tuple(values: Sequence[float], size: int) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64).reshape(size)
    return tuple(float(v) for v in arr)


def _list(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


@register_data
class CameraData(BaseData):
    type_id = "CameraSnapshot"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        position: Sequence[float] = (0.0, 0.0, 1.0),
        focal_point: Sequence[float] = (0.0, 0.0, 0.0),
        view_up: Sequence[float] = (0.0, 1.0, 0.0),
        clipping_range: Sequence[float] = (0.01, 1000.0),
        view_angle: float = 30.0,
        parallel_projection: bool = False,
        parallel_scale: float = 1.0,
        window_center: Sequence[float] = (0.0, 0.0),
        view_shear: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        super().__init__()
        self.name = name or f"Camera Snapshot {self.id}"
        self.position = _float_tuple(position, 3)
        self.focal_point = _float_tuple(focal_point, 3)
        self.view_up = _float_tuple(view_up, 3)
        self.clipping_range = _float_tuple(clipping_range, 2)
        self.view_angle = float(view_angle)
        self.parallel_projection = bool(parallel_projection)
        self.parallel_scale = float(parallel_scale)
        self.window_center = _float_tuple(window_center, 2)
        self.view_shear = _float_tuple(view_shear, 3)

    @classmethod
    def from_camera(cls, camera: vtk.vtkCamera, name: Optional[str] = None) -> "CameraData":
        obj = cls(name=name)
        obj.update_from_camera(camera)
        if name is None:
            obj.name = f"Camera Snapshot {obj.id}"
        return obj

    def update_from_camera(self, camera: vtk.vtkCamera) -> None:
        self.position = _float_tuple(camera.GetPosition(), 3)
        self.focal_point = _float_tuple(camera.GetFocalPoint(), 3)
        self.view_up = _float_tuple(camera.GetViewUp(), 3)
        self.clipping_range = _float_tuple(camera.GetClippingRange(), 2)
        self.view_angle = float(camera.GetViewAngle())
        self.parallel_projection = bool(camera.GetParallelProjection())
        self.parallel_scale = float(camera.GetParallelScale())
        self.window_center = _float_tuple(camera.GetWindowCenter(), 2)
        self.view_shear = _float_tuple(camera.GetViewShear(), 3)

    def apply_to_camera(self, camera: vtk.vtkCamera) -> None:
        camera.SetPosition(*self.position)
        camera.SetFocalPoint(*self.focal_point)
        camera.SetViewUp(*self.view_up)
        camera.SetClippingRange(*self.clipping_range)
        camera.SetViewAngle(self.view_angle)
        camera.SetParallelProjection(self.parallel_projection)
        camera.SetParallelScale(self.parallel_scale)
        camera.SetWindowCenter(*self.window_center)
        camera.SetViewShear(*self.view_shear)

    def get_extents(self) -> tuple[float, float, float, float, float, float] | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.get_type_id(),
            "name": self.name,
            "position": _list(self.position),
            "focal_point": _list(self.focal_point),
            "view_up": _list(self.view_up),
            "clipping_range": _list(self.clipping_range),
            "view_angle": float(self.view_angle),
            "parallel_projection": bool(self.parallel_projection),
            "parallel_scale": float(self.parallel_scale),
            "window_center": _list(self.window_center),
            "view_shear": _list(self.view_shear),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraData":
        type_id = data.get("type_id", cls.get_type_id())
        if type_id != cls.get_type_id():
            raise ValueError(f"Expected camera snapshot type_id '{cls.get_type_id()}', got '{type_id}'.")
        return cls(
            name=str(data.get("name", "")) or None,
            position=data.get("position", (0.0, 0.0, 1.0)),
            focal_point=data.get("focal_point", (0.0, 0.0, 0.0)),
            view_up=data.get("view_up", (0.0, 1.0, 0.0)),
            clipping_range=data.get("clipping_range", (0.01, 1000.0)),
            view_angle=float(data.get("view_angle", 30.0)),
            parallel_projection=bool(data.get("parallel_projection", False)),
            parallel_scale=float(data.get("parallel_scale", 1.0)),
            window_center=data.get("window_center", (0.0, 0.0)),
            view_shear=data.get("view_shear", (0.0, 0.0, 1.0)),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CameraData":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Camera snapshot JSON must contain a JSON object.")
        return cls.from_dict(raw)

    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id
        group.attrs["name"] = self.name
        group.attrs["view_angle"] = self.view_angle
        group.attrs["parallel_projection"] = int(self.parallel_projection)
        group.attrs["parallel_scale"] = self.parallel_scale
        group.create_dataset("position", data=np.asarray(self.position, dtype=np.float64))
        group.create_dataset("focal_point", data=np.asarray(self.focal_point, dtype=np.float64))
        group.create_dataset("view_up", data=np.asarray(self.view_up, dtype=np.float64))
        group.create_dataset("clipping_range", data=np.asarray(self.clipping_range, dtype=np.float64))
        group.create_dataset("window_center", data=np.asarray(self.window_center, dtype=np.float64))
        group.create_dataset("view_shear", data=np.asarray(self.view_shear, dtype=np.float64))
        return group

    @classmethod
    def load_hdf5(cls, group: h5py.Group) -> "CameraData":
        def dataset_tuple(name: str, size: int, fallback: Sequence[float]) -> tuple[float, ...]:
            ds = group.get(name)
            if isinstance(ds, h5py.Dataset):
                return _float_tuple(ds[()], size)
            return _float_tuple(fallback, size)

        obj = cls(
            name=_decode(group.attrs.get("name", "")) or None,
            position=dataset_tuple("position", 3, (0.0, 0.0, 1.0)),
            focal_point=dataset_tuple("focal_point", 3, (0.0, 0.0, 0.0)),
            view_up=dataset_tuple("view_up", 3, (0.0, 1.0, 0.0)),
            clipping_range=dataset_tuple("clipping_range", 2, (0.01, 1000.0)),
            view_angle=float(group.attrs.get("view_angle", 30.0)),
            parallel_projection=bool(group.attrs.get("parallel_projection", 0)),
            parallel_scale=float(group.attrs.get("parallel_scale", 1.0)),
            window_center=dataset_tuple("window_center", 2, (0.0, 0.0)),
            view_shear=dataset_tuple("view_shear", 3, (0.0, 0.0, 1.0)),
        )
        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = _decode(stored_id)
        if not obj.name:
            obj.name = f"Camera Snapshot {obj.id}"
        return obj
