from dataclasses import dataclass, field
from typing import Any, Protocol, ClassVar, Callable
from abc import ABC, abstractmethod


@dataclass
class Command(ABC):
    title: str
    @abstractmethod
    def execute(self) -> None: ...
    
    @abstractmethod
    def undo(self) -> None: ...
    
    
class LambdaCommand(Command):
    def __init__(self, label: str, execute: Callable, undo: Callable):
        super().__init__(label)
        self.execute = execute
        self.undo = undo
        

@dataclass
class CommandManager:
    undo_stack: list[Command] = field(default_factory=list)
    redo_stack: list[Command] = field(default_factory=list)
    
    
    def do(self, command: Command) -> None:
        print(f'doing command {command.title}')
        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()
        
    def undo(self) -> None:
        print(f'undo called. {len(self.undo_stack)=}')
        if not self.undo_stack:
            return
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)
        
    def redo(self) -> None:
        if not self.redo_stack:
            return
        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)
        
        
@dataclass
class CompositeCommand:
    label: str
    parts: list[Command]
    def execute(self) -> None:
        for cmd in self.parts:
            cmd.execute()
    
    def undo(self) -> None:
        for cmd in reversed(self.parts):
            cmd.undo()
            
                
# example

"""
@dataclass
class AddNodeCommand:
    label: str = "Add node"
    graph: Graph = ...
    node: Node = ...
    def execute(self): self.graph.add_node(self.node)
    def undo(self): self.graph.remove_node(self.node)

"""