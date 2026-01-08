from .base import CommandTool, ToolZone
from .tool_context import ToolContext

from typing import ClassVar, Optional, Sequence, cast
from dataclasses import dataclass

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton

import vtk
from geon.ui.toolbar import CommonToolsDock, CommonToolsWidget
from geon.util.resources import resource_path


def _get_active_actor(ctx: ToolContext) -> Optional[vtk.vtkProp]:
    scene = ctx.scene
    if scene is None:
        return None
    layer = scene.active_layer
    if layer is None:
        return None
    actors = layer.actors
    if not actors or len(actors) == 0:
        return None
    return actors[0]


def _apply_canonical_view(ctx: ToolContext, direction: Sequence[float], up: Sequence[float]) -> None:
    actor = _get_active_actor(ctx)
    if actor is None:
        return
    bounds = actor.GetBounds()
    if not bounds or len(bounds) != 6:
        return
    if bounds[0] > bounds[1] or bounds[2] > bounds[3] or bounds[4] > bounds[5]:
        return

    viewer = ctx.viewer
    center = (
        (bounds[0] + bounds[1]) * 0.5,
        (bounds[2] + bounds[3]) * 0.5,
        (bounds[4] + bounds[5]) * 0.5,
    )
    renderer = viewer._renderer
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(
        center[0] + direction[0],
        center[1] + direction[1],
        center[2] + direction[2],
    )
    camera.SetViewUp(*up)
    renderer.ResetCamera(bounds)
    renderer.ResetCameraClippingRange()
    viewer.rerender()


def _update_tool_icon(ctx: ToolContext, tooltip: str, icon_path: str) -> None:
    viewer = ctx.viewer
    if viewer is None:
        return
    window = viewer.window()
    if window is None:
        return
    try:
        tool_dock = cast(CommonToolsDock, window.tool_dock) # type:ignore
        tool_grid = cast(CommonToolsWidget, tool_dock.tool_grid)
        layout = tool_grid.grid_layout
    except AttributeError:
        return
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QPushButton) and widget.toolTip() == tooltip:
            widget.setIcon(QIcon(icon_path))


@dataclass
class TogglePerspectiveTool(CommandTool):
    """
    Toggle between perspective and isometric projection
    """
    label:      ClassVar = "toggle_perspective"
    tooltip:    ClassVar = "Toggle between perspective and isometric projection"
    icon_path:  ClassVar = resource_path("camera_perspective_toggle.png")
    shortcut:   ClassVar = "F3" # TODO: correct shortcut string?
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True
    
    def __post_init__(self):
        super().__post_init__()
        self._sync_icon_with_state()

    def _sync_icon_with_state(self) -> None:
        camera = self.ctx.viewer._renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            type(self).icon_path = resource_path("camera_isometric_toggle.png")
        else:
            type(self).icon_path = resource_path("camera_perspective_toggle.png")
        
        icon_path = TogglePerspectiveTool.icon_path
        if self.tooltip is not None and icon_path is not None:
            _update_tool_icon(self.ctx, self.tooltip, icon_path)
    
    def trigger(self) -> None:
        self.ctx.viewer.toggle_projection()
        self._sync_icon_with_state()
    
    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
        
        
@dataclass
class CameraTopTool(CommandTool):
    """
    Set camera to top view.
    """
    label:      ClassVar = "camera_top"
    tooltip:    ClassVar = "Set camera to top view."
    icon_path:  ClassVar = resource_path("camera_top.png")
    shortcut:   ClassVar = None
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True
        
    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0))
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
        
        
@dataclass
class CameraBottomTool(CommandTool):
    label:      ClassVar = "camera_bottom"
    tooltip:    ClassVar = "Set camera to bottom view."
    icon_path:  ClassVar = resource_path("camera_bot.png")
    shortcut:   ClassVar = ""
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True

    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))

    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()


@dataclass
class CameraLeftTool(CommandTool):
    label:      ClassVar = "camera_left"
    tooltip:    ClassVar = "Set camera to left view."
    icon_path:  ClassVar = resource_path("camera_left.png")
    shortcut:   ClassVar = None
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True

    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(-1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))

    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()


@dataclass
class CameraRightTool(CommandTool):
    label:      ClassVar = "camera_right"
    tooltip:    ClassVar = "Set camera to right view."
    icon_path:  ClassVar = resource_path("camera_right.png")
    shortcut:   ClassVar = None
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True

    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))

    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()


@dataclass
class CameraFrontTool(CommandTool):
    label:      ClassVar = "camera_front"
    tooltip:    ClassVar = "Set camera to front view."
    icon_path:  ClassVar = resource_path("camera_front.png")
    shortcut:   ClassVar = None
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True

    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0))

    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()


@dataclass
class CameraBackTool(CommandTool):
    label:      ClassVar = "camera_back"
    tooltip:    ClassVar = "Set camera to back view."
    icon_path:  ClassVar = resource_path("camera_back.png")
    shortcut:   ClassVar = None
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_VIEWPORT}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True

    def trigger(self) -> None:
        _apply_canonical_view(self.ctx, direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0))

    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
