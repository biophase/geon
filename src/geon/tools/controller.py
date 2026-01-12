from .base import BaseTool, ModeTool, CommandTool
from .tool_context import ToolContext
from .registry import TOOL_REGISTRY
from .command_manager import CommandManager, Command
from ..ui.context_ribbon import ContextRibbon
from ..rendering.base import BaseLayer

from typing import Optional

from PyQt6.QtCore import QEvent, pyqtSignal, QObject, Qt
from PyQt6.QtWidgets import (
    QWidget, QApplication, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, 
    QDoubleSpinBox, QComboBox
)
from PyQt6.QtGui import QShortcut, QKeySequence

class ToolController(QObject):
    # emitted on tool activation
    tool_activated              = pyqtSignal(QWidget)
    
    # emitted on tool deactivation
    tool_deactivated            = pyqtSignal()
    
    # emitted on changes in the internal selection state of a layer 
    # e.g. for UI updates
    layer_internal_sel_changed  = pyqtSignal(BaseLayer)
    
    scene_tree_request_change = pyqtSignal()
    
    # toolCaptureTelemetry        = pyqtSignal(BaseTool)
    
    def __init__(self, context_ribbon: ContextRibbon):
        super().__init__()
        self._ctx : Optional[ToolContext] = None 
        self._active_tool : Optional[BaseTool] = None
        self._global_command_manager: CommandManager = CommandManager()
        self._last_tool: Optional[BaseTool] = None
        
        self.ribbon : ContextRibbon = context_ribbon
        
    @property
    def ctx(self) -> Optional[ToolContext]:
        return self._ctx
        
    @ctx.setter
    def ctx(self, ctx: ToolContext) -> None:
        # potential cleanup code here
        self._ctx = ctx
        
    @property
    def command_manager(self) -> CommandManager:
        if self._active_tool is not None and self._active_tool.use_local_cm:
            return self._active_tool.command_manager
        else:
            return self._global_command_manager
    
    @property
    def is_active(self) -> bool:
        return self._active_tool is not None
    
    @property
    def active_tool_tooltip(self) -> str:
        if self._active_tool is not None:
            if self._active_tool.tooltip is None:
                return "No tooltip"
            else:
                return self._active_tool.tooltip
        else:
            return ('No active tool')
    
    def activate_tool(self, tool_id: str) -> None:
        try:
            if self.ctx is None:
                return
            if self._active_tool is not None:
                self._active_tool.deactivate()
                self._active_tool = None
                
            tool = TOOL_REGISTRY.create(
                tool_id, 
                self._global_command_manager, 
                self.ctx)
            
            self._active_tool = tool
            self._last_tool = tool
            
            if isinstance(tool, CommandTool):
                tool.trigger()
            else:
                tool.activate()
                
            if isinstance(tool, ModeTool):
                w = QWidget(self.ribbon)
                self.tool_activated\
                    .emit(tool.create_context_widget(parent=w))

                
                    
        except Exception as e:
            print(e)
            
    
    def deactivate_tool(self) -> None:
        if self._active_tool is None:
            return
        tool = self._active_tool
        self._active_tool.deactivate()
        self.ribbon.clear_group('tool')
        self._active_tool = None
        self._last_tool = tool
        self.tool_deactivated.emit()
    
    def do_global(self, cmd: Command) -> None:
        self._global_command_manager.do(cmd)
    
    @property
    def active_tool(self) -> Optional[BaseTool]:
        if self._active_tool is not None:
            return self._active_tool

    @property
    def last_tool(self) -> Optional[BaseTool]:
        return self._last_tool
        
    def install_tool_schortcuts(self, parent: QWidget) -> None:
        for tool_id,  tool_cls in TOOL_REGISTRY._tools.items():
            shortcut = tool_cls.shortcut
            if shortcut is None:
                continue
            
            sc = QShortcut(QKeySequence(shortcut), parent)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(lambda tid=tool_id: self._run_shortcut(tid))
            
    def _run_shortcut(self, tool_id: str) -> None:
        w = QApplication.focusWidget()
        
        # ignore shortcuts if focus is on:
        if isinstance(w, (
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
            QComboBox
            )):
            return
        self.activate_tool(tool_id)

    def clear_undo_stacks(self) -> None:
        """
        Clear undo/redo stacks for global and active tool managers.
        """
        self._global_command_manager.undo_stack.clear()
        self._global_command_manager.redo_stack.clear()
        if self._active_tool is not None and self._active_tool.use_local_cm:
            cm = self._active_tool.command_manager
            cm.undo_stack.clear()
            cm.redo_stack.clear()
        
