from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Literal, ClassVar


# from geon.io.dataset import DocumentReference
from geon.rendering.scene import Scene

from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt

from .command_manager import CommandManager, Command
from .tool_context import ToolContext


class ToolZone(Enum):
    SIDEBAR_RIGHT_ESSENTIALS    = auto()
    SIDEBAR_RIGHT_VIEWPORT      = auto()
    
class CameraMode(Enum):
    DEFAULT     = auto()
    DISABLED    = auto()
    
    
@dataclass
class Event:
    pos: tuple[int,int]
    prev_pos: tuple[int,int]
    shift: bool
    ctrl: bool
    alt: bool
    key: Optional[str]

    
@dataclass
class BaseTool (ABC):

    label:      ClassVar[Optional[str]]
    tooltip:    ClassVar[Optional[str]]
    icon_path:  ClassVar[Optional[str]]
    shortcut:   ClassVar[Optional[str]]
    ui_zones:   ClassVar[set[ToolZone]]
    use_local_cm:       ClassVar[bool] = False
    show_in_toolbar:    ClassVar[bool] = True
    

    command_manager: CommandManager
    ctx: ToolContext
    _in_use: bool = field(init=False)
    
    def __post_init__(self):
        self._in_use=False


    @abstractmethod
    def activate(self) -> None:
        """
        Tool activation, cursor changes, observer registration, etc.
        """
        self._in_use = True
        

    @abstractmethod
    def deactivate(self) -> None:
        """
        Clean up method.\n
        To use call `ToolController.deactivate_tool()`
        """
        self._in_use = False
        
  
    
@dataclass
class CommandTool(BaseTool):
    """
    for single shot commands
    sub classes should implement `trigger` rather than `activate`
    """
    @abstractmethod
    def trigger(self) -> None:
        """
        main method to override for single-command tools
        """
        ...
        
    def activate(self) -> None:
        super().activate()
        self.trigger()
        self.ctx.controller.deactivate_tool()

@dataclass
class InitModeToolCmd(Command):
    """
    Proxy command put at the beginning of tool-internal undo stacks.\n
    Undoing this command exits the interactive tool.
    """
    tool: BaseTool
    
    def execute(self) -> None:
        pass
    def undo(self) -> None:
        self.tool.ctx.controller.deactivate_tool()
        
@dataclass
class ModeTool(BaseTool):
    """
    A tool that stays active and hooks into mouse / key events
    """
    cursor_icon_path : ClassVar[Optional[str]] = None
    pixmap: Optional[QPixmap] = None
    cursor: Optional[QCursor] = None
    cursor_hot: ClassVar[Optional[tuple[int,int]]] = None
    keep_focus: ClassVar[bool] = True


    def __post_init__(self):
        if self.cursor_icon_path is not None:
            pixmap = QPixmap(self.cursor_icon_path)
            pixmap = pixmap.scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
            self.pixmap = pixmap
            hot = self.cursor_hot
            if hot is not None:
                self.cursor = QCursor(self.pixmap, hot[0], hot[1])
            else:
                self.cursor = QCursor(self.pixmap, 0, 0)
            
        if self.use_local_cm:
            print(f'local command manager init for {__class__.__name__}')
            self.command_manager = CommandManager()
            init_tool_cmd = InitModeToolCmd(f'Initialized {self.tooltip}', self)
            self.command_manager.do(init_tool_cmd)

    @abstractmethod
    def activate(self) -> None:
        super().activate()
        if self.cursor is not None:
            QApplication.setOverrideCursor(self.cursor)
            
    def deactivate(self) -> None:
        if self.cursor is not None:
            QApplication.restoreOverrideCursor()
        return super().deactivate()
            

    def left_button_press_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on left mouse click
        """
        pass
    
    def double_click_press_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on left mouse click
        """
        pass
            
    def right_button_press_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on right mouse click
        """
        pass
    
    def middle_button_press_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on mouse wheel click
        """
        pass
    
    def mouse_wheel_forward_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on mouse wheel forward
        """
        pass
    
    def mouse_wheel_backward_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on mouse wheel backward
        """
        pass
    
    def mouse_move_event_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on mouse movement 
        """
        pass

    def key_press_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on a keyboard key press
        """
        pass
    
    def key_release_hook(self, event: Event) -> None:
        """
        override to define tool behaviour on a keyboard key press
        """
        pass
    
    # context UI
    def create_context_widget(self, parent: QWidget) -> QWidget | None:
        """
        Creates the context widget for the tool
        """
        pass
