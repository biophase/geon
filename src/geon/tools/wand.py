from geon.tools.base import Event
from config.theme import UIStyle
from .base import ModeTool, ToolZone
from .selection import SelectPointsCmd
from ..rendering.pointcloud import PointCloudLayer


from dataclasses import dataclass, field
import sys
from typing import ClassVar, Optional
import weakref

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QDoubleSpinBox

import numpy as np
from geon.util.resources import resource_path


@dataclass
class WandTool(ModeTool):
    # general settings
    label: ClassVar = 'wand'
    tooltip: ClassVar = "Magic wand tool"
    icon_path: ClassVar = resource_path('wand.png')
    shortcut: ClassVar = 'w'
    ui_zones: ClassVar = {ToolZone.SIDEBAR_RIGHT_ESSENTIALS}
    use_local_cm: ClassVar[bool] = False
    show_in_toolbar: ClassVar[bool] = True
    cursor_icon_path : ClassVar = resource_path('wand.png')
    cursor_hot: ClassVar = (3, 3) 
    
    # mode tool settings
    keep_focus: ClassVar[bool] = False
    
    # state
    tolerance: float = 0.5
    
    def _on_click(self):
        result = self.ctx.viewer.pick()
        if result.layer is None:
            return
        
        if result.layer.id == self.ctx.scene.active_layer_id:
            if isinstance(result.layer, PointCloudLayer):
                af = result.layer.active_field
                idx = result.element_idx
                if idx is None or af is None:
                    return
                data = af.data
                picked_data = np.asarray(data[idx])
                
                print(f"[wand] {picked_data.shape=}")
                print(f"[wand] {picked_data.ndim=}")
                
                
                
                if picked_data.ndim == 0:
                    similarity = data - picked_data
                elif picked_data.ndim == 1:
                    similarity = np.linalg.norm(data - picked_data, axis=1)
                    
                else:
                    raise NotImplementedError(f"Unexpected data shape of picked point: {picked_data.shape}")
                print(f"[wand] {similarity.shape=}")
                in_tol = np.nonzero(np.abs(similarity) < self.tolerance)[0]
                print(f"[wand] selected elements {np.sum(in_tol)}")
                cmd = SelectPointsCmd(
                    title="Wand selection",
                    selection_new=in_tol,
                    layer_ref=weakref.ref(result.layer),
                    ctx_ref=weakref.ref(self.ctx)
                )
                self.command_manager.do(cmd)
                
                
            else:
                raise NotImplementedError(f"No wand implementation for {type(result.layer)}")
            
            
    
    # ------------------------------------------------
    # hooks
    # ------------------------------------------------
    
    def left_button_press_hook(self, event: Event) -> None:
        self._on_click()
        super().left_button_press_hook(event)
        
    def activate(self) -> None:
        return super().activate()
    
    def deactivate(self) -> None:
        return super().deactivate()
    
    def create_context_widget(self, parent: QWidget) -> QWidget | None:
        w = QWidget(parent)
        outer = QHBoxLayout(w)
        outer.setContentsMargins(2, 1, 2, 1)
        outer.setSpacing(2)
        tolerance_label = QLabel("tolerance: ")
        tolerance_label.setStyleSheet(UIStyle.TYPE_LABEL.value)
        outer.addWidget(tolerance_label)
        tolerance_input = QDoubleSpinBox(w)
        tolerance_input.setDecimals(3)
        tolerance_input.setSingleStep(1.0)
        tolerance_input.setRange(-sys.float_info.max, sys.float_info.max)
        tolerance_input.setValue(float(self.tolerance))
        tolerance_input.valueChanged.connect(lambda val: setattr(self, "tolerance", float(val)))
        outer.addWidget(tolerance_input)
        return w
    
    def key_press_hook(self, event: Event) -> None:
        super().key_press_hook(event)
        print(f"[wand] press event: {event.key}")
        if event.key is None:
            return
        if event.key.lower() == 'escape':
            self.ctx.controller.deactivate_tool()
            return
