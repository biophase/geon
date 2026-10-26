from PyQt6.QtWidgets import QLabel

from geon.rendering.mesh import MeshLayer
from .registry import LAYER_UI, LayerUIHooks


def _ribbon(layer, parent, controller):
    return QLabel(f"Vertices: {len(layer.data.vertices):,}    Faces: {len(layer.data.faces):,}", parent)


LAYER_UI.register(MeshLayer, LayerUIHooks(
    ribbon_widget=_ribbon,
    tree_item_text=lambda layer: layer.browser_name,
))
