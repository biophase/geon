from __future__ import annotations

import vtk

from geon.data.camera import CameraData

from .base import BaseLayer
from .layer_registry import layer_for


@layer_for(CameraData)
class CameraLayer(BaseLayer[CameraData]):
    @classmethod
    def merge(cls, layers, *, name=None):
        raise ValueError('Camera snapshots cannot be merged.')

    layer_type_id = "camera_snapshot"

    def _build_pipeline(
        self,
        renderer: vtk.vtkRenderer,
        out_actors: list[vtk.vtkProp],
    ) -> None:
        return None

    def update(self) -> None:
        pass

    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.data.focal_point)

    def data_index_from_picked_id(self, sub_id: int) -> int:
        return -1
