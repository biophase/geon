from .base import CommandTool, ToolZone
from .command_manager import Command
from .tool_context import ToolContext
from .registry import TOOL_REGISTRY
from ..rendering.base import BaseLayer
from ..rendering.pointcloud import PointCloudLayer

from dataclasses import dataclass, field
from typing import ClassVar, Optional
import weakref

import numpy as np
from geon.util.resources import resource_path



@dataclass
class SelectPointsCmd(Command):
    """
    Applies a reversible selection to a point cloud layer.
    `selection_old` allows passin an explicit old selection, otherwise 
    infer from layer
    """
    selection_new: Optional[np.ndarray]
    layer_ref: weakref.ReferenceType[PointCloudLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    selection_old: Optional[np.ndarray] = None 
    
    def execute(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        if self.selection_old is None:
            self.selection_old = (
                layer.active_selection.copy()
                if layer.active_selection is not None
                else None
            )
        if self.selection_new is None:
            layer.active_selection = None
        else:
            layer.active_selection = self.selection_new
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()
        
    def undo(self) -> None:
        layer = self.layer_ref()
        ctx = self.ctx_ref()
        if layer is None or ctx is None:
            return
        layer.active_selection = self.selection_old
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()
        

@dataclass
class DeselectTool(CommandTool):
    label:      ClassVar = "deselect"
    tooltip:    ClassVar = "Deselect"
    icon_path:  ClassVar = resource_path("deselect.png")
    shortcut:   ClassVar = "Ctrl+D"
    ui_zones:   ClassVar = {
        ToolZone.SIDEBAR_RIGHT_ESSENTIALS,
        }
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
                cmd = SelectPointsCmd(
                    title="Deselect points",
                    selection_new=None,
                    layer_ref=weakref.ref(layer),
                    ctx_ref=weakref.ref(self.ctx),
                    selection_old=layer.active_selection
                )
                self.command_manager.do(cmd)
            
        else:
            raise NotImplementedError
        

    
    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
    
    
