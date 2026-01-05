from PyQt6.QtWidgets import (QDockWidget, QWidget, QGridLayout, QPushButton, QFrame)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSize

from ..tools.registry import TOOL_REGISTRY
from ..tools.base import ToolZone
from ..tools.controller import ToolController

from typing import Optional, cast

# --- GLOBAL SETTINGS ---
# Define the fixed, rectangular size for all tool buttons
TOOL_BUTTON_SIZE = QSize(36, 36)
# Define the fixed spacing/margins for the grid
GRID_SPACING = 0
GRID_MARGIN = 0
# -----------------------

class CommonToolsDock(QDockWidget):
    """
    Toolgrid for global tools (e.g. for camera control, visual settings, etc.)
    """
    def __init__(self, 
                 title: str, 
                 parent=None,
                 controller: Optional[ToolController] = None
                 ):
        super().__init__(title, parent)
        self.setObjectName(title.replace(" ", "_"))
        self.tool_grid = CommonToolsWidget(self, controller=controller)
        self.setWidget(self.tool_grid)



class CommonToolsWidget(QWidget):

    def __init__(self, 
                 parent=None,
                 controller: Optional[ToolController] = None
                 ):
        super().__init__(parent)

        self.controller = controller
        self.shows_ui_zones: set[ToolZone] = {
            ToolZone.SIDEBAR_RIGHT_ESSENTIALS,
            ToolZone.SIDEBAR_RIGHT_VIEWPORT
            }
        
        
        icon_size = QSize(32,32)    
        self.max_cols: int = 1
        zone_list = sorted(
            self.shows_ui_zones,
            key=lambda zone: getattr(zone, "value", str(zone)),
        )
        grid_width = (
            (self.max_cols * TOOL_BUTTON_SIZE.width())
            + (GRID_SPACING * (self.max_cols - 1))
            + (GRID_MARGIN * 2)
        )
        self.setFixedWidth(grid_width)

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(GRID_MARGIN, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN)
        self.grid_layout.setSpacing(GRID_SPACING)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # dummy_tool_list = [
        #     ("resources/load.png", "Load PointCloud"),
        #     ("resources/settings.png", "Settings"),
        #     ("resources/increase_point_size.png", "Increase point size"),
        #     ("resources/decrease_point_size.png", "Decrease point size"),
        #     ("resources/reset_view.png", "Reset View"),
        #     ("resources/lasso.png", "Lasso segmentation"),
        #     ("resources/label_semantics.png", "Label semantics"),
        #     ("resources/label_instance.png", "Label instance"),
        #     ("resources/wand.png", "Wand tool"),
        #     ("resources/deselect.png", "Deselect selection"),
        #     ("resources/hide_selected.png", "Hide selected points"),
        #     ("resources/show_all.png", "Show all points"),
        #     ("resources/reggrow.png", "Start regioin growing"),
        #     ("resources/reggrow_settings.png", "Region growing settings"),
        #     ("resources/create_graph.png", "Create a graph"),
        #     ("resources/interactive_segmentation.png", "Interactive segmentation"),
        # ]
        row = 0
        for zone_idx, zone in enumerate(zone_list):
            zone_tools = TOOL_REGISTRY.get_zone_tools(zone)
            zone_res_list = [(c.icon_path, c.tooltip, c.__name__) for c in zone_tools                 if c.show_in_toolbar]

            col = 0
            for icon_path, tooltip, tool_id in zone_res_list:
                btn = QPushButton()
                btn.setFlat(True)
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(icon_size)
                btn.setFixedSize(TOOL_BUTTON_SIZE)
                btn.setToolTip(tooltip)
                def _safe_activate(tid: str) -> None:
                    # print('activate',tid, 'ctx is', self.controller.ctx)
                    assert self.controller is not None
                    self.controller.activate_tool(tid)


                if self.controller is not None:
                    btn.pressed.connect(lambda tid=tool_id: _safe_activate(tid))

                self.grid_layout.addWidget(btn, row, col)
                col +=1 
                if col >= self.max_cols:
                    col = 0
                    row += 1
            if col != 0:
                row += 1
            if zone_idx < (len(zone_list) - 1):
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                self.grid_layout.addWidget(separator, row, 0, 1, self.max_cols)
                row += 1
            
            






    
