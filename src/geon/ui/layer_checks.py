"""Active-layer checks shared by actions for different data types."""

from typing import TypeVar

from PyQt6.QtWidgets import QMessageBox, QWidget

from ..rendering.boundingbox import BoundingBoxLayer
from ..rendering.cellcomplex import CellComplexLayer
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.scene import Scene


LayerT = TypeVar("LayerT", PointCloudLayer, BoundingBoxLayer, CellComplexLayer)

_LAYER_NAMES = {
    PointCloudLayer: "point cloud",
    BoundingBoxLayer: "bounding box",
    CellComplexLayer: "cell complex",
}


def require_active_layer(
    parent: QWidget, scene: Scene | None, layer_type: type[LayerT]
) -> LayerT | None:
    """Return the active layer, or show an error if its type does not match."""
    layer = scene.active_layer if scene is not None else None
    if isinstance(layer, layer_type):
        return layer
    QMessageBox.critical(
        parent, "Active layer required", f"Activate a {_LAYER_NAMES[layer_type]} first."
    )
    return None
