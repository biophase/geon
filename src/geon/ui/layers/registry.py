from typing import Type, Optional, Callable, TypeVar, Generic
from ...rendering.base import BaseLayer
from ...tools.controller import ToolController
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (QWidget, QMenu)
from PyQt6.QtGui import QIcon


T = TypeVar('T', bound=BaseLayer)
@dataclass
class LayerUIHooks(Generic[T]):
    ribbon_widget: Optional[Callable[[T, QWidget, ToolController], QWidget | None]] = None
    ribbon_sel_widget: Optional[Callable[[T, QWidget, ToolController], QWidget | None]] = None
    tree_menu: Optional[Callable[[T, QWidget, ToolController], QMenu]] = None
    tree_item_text: Optional[Callable[[T], str]] = None
    tree_item_icon: Optional[Callable[[T], QIcon]] = None
    

class LayerUIRegistry:
    def __init__(self) -> None:
        self._hooks: dict[Type[BaseLayer], LayerUIHooks] = dict()
         
    def register(self, layer_cls: Type[BaseLayer], hooks: LayerUIHooks) -> None:
         self._hooks[layer_cls] = hooks
         
    def resolve(self, layer: BaseLayer) -> LayerUIHooks:
        for cls in type(layer).mro():
            if cls in self._hooks:
                return self._hooks[cls]
        return LayerUIHooks()
     
LAYER_UI = LayerUIRegistry()