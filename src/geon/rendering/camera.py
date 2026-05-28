from __future__ import annotations

import vtk

from geon.data.camera import CameraData

from .base import BaseLayer
from .layer_registry import layer_for


@layer_for(CameraData)
class CameraLayer(BaseLayer[CameraData]):
    layer_type_id = "camera_snapshot"

    def __init__(self, data: CameraData):
        super().__init__(data)
        self.browser_name = data.name

    @property
    def browser_name(self) -> str:
        return self.data.name

    @browser_name.setter
    def browser_name(self, browser_name: str) -> None:
        self.data.name = browser_name
        self._browser_name = browser_name

    def _build_pipeline(
        self,
        renderer: vtk.vtkRenderer,
        out_actors: list[vtk.vtkProp],
    ) -> None:
        return None

    def update(self) -> None:
        self._browser_name = self.data.name

    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.data.focal_point)

    def data_index_from_picked_id(self, sub_id: int) -> int:
        return -1
