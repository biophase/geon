
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..rendering.scene import Scene
    from ..ui.viewer import VTKViewer
    from ..tools.controller import ToolController



@dataclass
class ToolContext:
    scene: "Scene"
    viewer: "VTKViewer"
    controller: "ToolController"