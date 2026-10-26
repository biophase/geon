from .base import CommandTool, ToolZone
from .command_manager import Command
from .tool_context import ToolContext
from .registry import TOOL_REGISTRY
from ..rendering.base import BaseLayer
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.selection import SelectableLayer

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional
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
    selection_new: Optional[Any]
    layer_ref: weakref.ReferenceType[BaseLayer]
    ctx_ref: weakref.ReferenceType[ToolContext]
    selection_old: Optional[Any] = None 
    
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
            layer.clear_selection()
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
        

class ClearSelectionCmd(Command):
    """Clear all selection state while retaining a complete undo snapshot."""
    def __init__(self, layer, ctx):
        super().__init__("Deselect selection")
        self.layer_ref = weakref.ref(layer)
        self.ctx_ref = weakref.ref(ctx)
        self.snapshot = layer.capture_selection()

    def _apply(self, restore):
        layer, ctx = self.layer_ref(), self.ctx_ref()
        if layer is None or ctx is None:
            return
        if restore:
            layer.restore_selection(self.snapshot)
        else:
            layer.clear_selection()
        ctx.controller.layer_internal_sel_changed.emit(layer)
        ctx.viewer.rerender()

    def execute(self):
        self._apply(False)

    def undo(self):
        self._apply(True)


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
        if isinstance(layer, SelectableLayer):
            self.command_manager.do(ClearSelectionCmd(layer, self.ctx))
        

    
    
    def deactivate(self) -> None:
        super().deactivate()
        self.ctx.viewer.rerender()
    
    
