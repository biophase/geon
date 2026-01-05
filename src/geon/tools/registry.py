from typing import Type, TypeVar
from .base import BaseTool, ToolZone

from .command_manager import CommandManager
from .tool_context import ToolContext
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Type[BaseTool]] = {}
        
    
    T = TypeVar("T", bound=BaseTool)
    def register(self, tool_cls: Type[T]) -> Type[T]:
        if tool_cls.__name__ in self._tools.keys():
            raise ValueError(f"Tool id {tool_cls.__name__} already registered")
        
        self._tools[tool_cls.__name__] = tool_cls
        return tool_cls
    
    
    def create(self, tool_id:str, command_manager: CommandManager, ctx:ToolContext) -> BaseTool:

        tool = self._tools[tool_id](command_manager,ctx)

        return tool
    
    def get_zone_tools(self, zone: ToolZone) -> list[Type[BaseTool]]:
        zone_tools = []
        for tool_id, tool_cls in self._tools.items():
            if zone in tool_cls.ui_zones:
                zone_tools.append(tool_cls)
                
        return zone_tools
    
TOOL_REGISTRY = ToolRegistry()