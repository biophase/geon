from geon.tools.base import Event
from geon.config.theme import UIStyle
from .base import ModeTool, ToolZone
from .selection import SelectPointsCmd
from ..rendering.pointcloud import PointCloudLayer
from ..core.constants import Boolean
from ..ui.boolean_dialog import BooleanChoiceDialog
from ..util.common import bool_op_index_mask


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
    
    def _combine_selection(
        self,
        layer: PointCloudLayer,
        selection_new: np.ndarray,
        event: Event,
    ) -> np.ndarray | None:
        selection_old = layer.active_selection
        if selection_old is None or selection_old.size == 0:
            return np.asarray(selection_new, dtype=np.int32)

        if event.ctrl:
            bool_op = Boolean.DIFFERENCE
        elif event.shift:
            bool_op = Boolean.UNION
        else:
            dlg = BooleanChoiceDialog(
                parent=self.ctx.viewer.window(),
                message="Choose how to combine with previous selection:",
            )
            dlg.exec()
            if dlg.choice is None:
                return None
            bool_op = dlg.choice

        return bool_op_index_mask(selection_old, selection_new, bool_op)

    def _on_click(self, event: Event):
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
                visible_inds = result.layer.visible_inds
                visible_data = data[visible_inds]
                picked_data = np.asarray(data[idx])

                if picked_data.ndim == 0:
                    similarity = visible_data - picked_data
                elif picked_data.ndim == 1:
                    similarity = np.linalg.norm(visible_data - picked_data, axis=1)
                else:
                    raise NotImplementedError(f"Unexpected data shape of picked point: {picked_data.shape}")
                in_tol = np.nonzero(np.abs(similarity) < self.tolerance)[0]
                selection_new = np.asarray(visible_inds[in_tol], dtype=np.int32)
                selection_combined = self._combine_selection(result.layer, selection_new, event)
                if selection_combined is None:
                    return
                cmd = SelectPointsCmd(
                    title="Wand selection",
                    selection_new=selection_combined,
                    layer_ref=weakref.ref(result.layer),
                    ctx_ref=weakref.ref(self.ctx),
                    selection_old=(
                        result.layer.active_selection.copy()
                        if result.layer.active_selection is not None
                        else None
                    ),
                )
                self.command_manager.do(cmd)
                
                
            else:
                raise NotImplementedError(f"No wand implementation for {type(result.layer)}")
            
            
    
    # ------------------------------------------------
    # hooks
    # ------------------------------------------------
    
    def left_button_press_hook(self, event: Event) -> None:
        self._on_click(event)
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
