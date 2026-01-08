
from .base import CommandTool
from .command_manager import Command
from .tool_context import ToolContext
from .base import ToolZone
from ..rendering.pointcloud import PointCloudLayer

from dataclasses import dataclass, field
from typing import ClassVar, Optional
import weakref

import numpy as np
from geon.util.resources import resource_path

@dataclass
class SetPointsVisibilityCmd(Command):
    """
    Applies a visibility mask command on a point cloud's points.
    """
    
    visibility_mask_old: Optional[np.ndarray] = field(init=False)
    visibility_mask_new: Optional[np.ndarray]
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    
    
    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        if layer._visibility_mask is not None:
            self.visibility_mask_old = layer._visibility_mask.copy()
        else:
            self.visibility_mask_old=None
        layer.set_visibility_mask(self.visibility_mask_new)
        ctx.viewer.rerender()
        
    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.set_visibility_mask(self.visibility_mask_old)
        ctx.viewer.rerender()
    

@dataclass
class HideAndDeselectCmd(Command):
    visibility_mask_old: Optional[np.ndarray] = field(init=False)
    selection_old: Optional[np.ndarray] = field(init=False)
    visibility_mask_new: Optional[np.ndarray]
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]

    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        self.visibility_mask_old = layer._visibility_mask
        self.selection_old = layer.active_selection
        layer.set_visibility_mask(self.visibility_mask_new)
        layer.active_selection = None
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()

    def undo(self) -> None: # FIXME: undo doesn't work as expected
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.set_visibility_mask(self.visibility_mask_old)
        layer.active_selection = self.selection_old
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()


@dataclass
class HideTool(CommandTool):
    """
    Uses the layer's active selection to set a visibility mask
    """
    label:      ClassVar = "hide"
    tooltip:    ClassVar = "Hide"
    icon_path:  ClassVar = resource_path("hide.png")
    shortcut:   ClassVar = "Ctrl+H"
    ui_zones:   ClassVar = set()
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True
    
    def trigger(self) -> None:
        layer = self.ctx.scene.active_layer
        if layer is None:
            return
        if isinstance(layer, PointCloudLayer):
            if layer.active_selection is None:
                return
            else:
                mask = None
                if layer._visibility_mask is None:
                    mask = np.ones(layer.data.points.shape[0],dtype=np.bool_)
                else:
                    mask = layer._visibility_mask.copy()
                mask[layer.active_selection] = 0
                cmd = HideAndDeselectCmd(
                    title="Hide points",
                    visibility_mask_new=mask,
                    layer_ref=weakref.ref(layer),
                    ctx_ref=weakref.ref(self.ctx)
                )
                self.command_manager.do(cmd)
            
        else:
            raise NotImplementedError
        
    
    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
        
@dataclass
class IsolateTool(CommandTool):
    """
    Uses the layer's active selection to set a visibility mask
    """
    label:      ClassVar = "isolate"
    tooltip:    ClassVar = "Isolate"
    icon_path:  ClassVar = resource_path("isolate.png")
    shortcut:   ClassVar = "Ctrl+I"
    ui_zones:   ClassVar = set()
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True
    
    def trigger(self) -> None:
        layer = self.ctx.scene.active_layer
        if layer is None:
            return
        if isinstance(layer, PointCloudLayer):
            if layer.active_selection is None:
                return
            else:
                mask = np.zeros(layer.data.points.shape[0],dtype=np.bool_)
                mask[layer.active_selection] = 1
                cmd = HideAndDeselectCmd(
                    title="Isolate points",
                    visibility_mask_new=mask,
                    layer_ref=weakref.ref(layer),
                    ctx_ref=weakref.ref(self.ctx)
                )
                self.command_manager.do(cmd)
            
        else:
            raise NotImplementedError

    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
        
        
@dataclass
class ShowTool(CommandTool):
    """
    Uses the layer's active selection to set a visibility mask
    """
    label:      ClassVar = "show_all"
    tooltip:    ClassVar = "Show all"
    icon_path:  ClassVar = resource_path("show_all.png")
    shortcut:   ClassVar = "Ctrl+O"
    ui_zones:   ClassVar = {ToolZone.SIDEBAR_RIGHT_ESSENTIALS}
    use_local_cm:    ClassVar = False
    show_in_toolbar: ClassVar = True
    
    def trigger(self) -> None:
        layer = self.ctx.scene.active_layer
        if layer is None:
            return
        if isinstance(layer, PointCloudLayer):

            cmd = SetPointsVisibilityCmd(
                title="Show all points",
                visibility_mask_new=None,
                layer_ref=weakref.ref(layer),
                ctx_ref=weakref.ref(self.ctx)
            )
            self.command_manager.do(cmd)
            
        else:
            raise NotImplementedError
    
    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
