from .registry import ToolRegistry, TOOL_REGISTRY

"""
NOTE: tool imports at module level are essential for populating the registry
and toolbars
"""
from .lasso import LassoTool
from .selection import DeselectTool
from .visibility import HideTool, IsolateTool, ShowTool
from .annotate import AnnotateTool
from .wand import WandTool
from .seeded_region_growing import SeededRegionGrowingTool
from .inspect import InspectTool
from .cellcomplex import (
    CellComplexAddEdgeTool,
    CellComplexAnnotateTool,
    CellComplexAssociateTool,
    CellPickVertexTool,
)
from .camera import (
    TogglePerspectiveTool,
    CameraTopTool,
    CameraBottomTool,
    CameraLeftTool,
    CameraRightTool,
    CameraFrontTool,
    CameraBackTool,
)

# NOTE: Tool register order determines order in ui tool zones
TOOL_REGISTRY.register(InspectTool)
TOOL_REGISTRY.register(LassoTool)
TOOL_REGISTRY.register(WandTool)
TOOL_REGISTRY.register(SeededRegionGrowingTool)
TOOL_REGISTRY.register(DeselectTool)
TOOL_REGISTRY.register(HideTool)
TOOL_REGISTRY.register(IsolateTool)
TOOL_REGISTRY.register(ShowTool)
TOOL_REGISTRY.register(AnnotateTool)
TOOL_REGISTRY.register(CellPickVertexTool)
TOOL_REGISTRY.register(CellComplexAddEdgeTool)
TOOL_REGISTRY.register(CellComplexAssociateTool)
TOOL_REGISTRY.register(CellComplexAnnotateTool)
TOOL_REGISTRY.register(TogglePerspectiveTool)
TOOL_REGISTRY.register(CameraTopTool)
TOOL_REGISTRY.register(CameraBottomTool)
TOOL_REGISTRY.register(CameraLeftTool)
TOOL_REGISTRY.register(CameraRightTool)
TOOL_REGISTRY.register(CameraFrontTool)
TOOL_REGISTRY.register(CameraBackTool)
