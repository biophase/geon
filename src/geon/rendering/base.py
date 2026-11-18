from abc import ABC, abstractmethod
import vtk
from geon.data.base import BaseData

from typing import Any, TypeVar, Generic, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from PyQt6.QtCore import QObject, pyqtSignal


class LayerSignals(QObject):
    selectionChanged = pyqtSignal()
    pointVisibilityChanged = pyqtSignal()
    fieldDataChanged = pyqtSignal()
    activeFieldChanged = pyqtSignal()
    colorDisplayChanged = pyqtSignal()

TData = TypeVar("TData", bound=BaseData) 




@dataclass
class BaseLayer(Generic[TData], ABC):
    """
    Abstract base for all renderable layers.
    
    A layer:
        - wraps a BAseData instance (e.g. PointCloudData, ...)
        - owns the VTK pipeline objects needed to render it
        - can be attatched to a vtkRenderer

    """


    data: TData
    signals: LayerSignals = field(default_factory=LayerSignals, init=False, repr=False, compare=False)
    visible: bool = True
    # active: bool = True
    _browser_name: str = 'Untitled'

    
    # VTK
    _renderer: Optional[vtk.vtkRenderer] = field(default=None, init=False, repr=False)
    _actors: list[vtk.vtkProp] = field(default_factory= list, init=False, repr=False)

    @classmethod
    @abstractmethod
    def merge(cls, layers, *, name=None):
        """Construct an unattached merged layer; never edit the source layers."""
        raise NotImplementedError

    @classmethod
    def can_merge(cls, layers):
        return (len(layers)>=2 and all(type(layer) is cls for layer in layers)
                and len({id(layer) for layer in layers})==len(layers)
                and len({type(layer.data) for layer in layers})==1
                and all(layer.data.supports_merge and not getattr(layer,'is_preview',False) for layer in layers))

    @classmethod
    def _merge_layers(cls, layers, name):
        layers=list(layers)
        if not cls.can_merge(layers):
            raise ValueError('Select at least two mergeable layers of the same type.')
        first=layers[0]
        name=name or first._browser_name
        data=type(first.data).merge([layer.data for layer in layers],name=name)
        merged=cls(data)
        merged.browser_name=name
        merged.visible=first.visible
        return merged,first
    
    def attach(self, renderer: vtk.vtkRenderer) -> None:
        """
        Attach layer to a VTK Renderer
        
        This will:
            - build the pipeline
            - add the resulting actors to the renderer
        """
        
        if self._renderer is not None:
            raise RuntimeError("Layer is already attatched to a renderer.")
        
        self._renderer = renderer
        self._actors.clear()
        
        # create actors in subclass!
        self._build_pipeline(renderer, self._actors)
        
        for actor in self._actors:
            renderer.AddActor(actor)
            
    def detach(self) -> None:
        
        if self._renderer is None:
            return
        for actor in self._actors:
            self._renderer.RemoveActor(actor)
            
        self._actors.clear()
        self._renderer = None
        self.on_detached()
        
        
    # abstract hooks
    
    @abstractmethod
    def _build_pipeline(
        self,
        renderer: vtk.vtkRenderer,
        out_actors: list[vtk.vtkProp]
    ) -> None:
        """
        Subclasses build their pipelie here
        """
        ...
    @abstractmethod
    def update(self) -> None:
        """
        Called when the data has changed and the layer should update its VTK pipeline
        e.g. colors, geometry ...
        
        """
        ...
    @abstractmethod
    def world_xyz_from_picked_id(self, sub_id: int) -> tuple[float,float,float]:
        """
        Called e.g. for setting a 3d camera pivot
        """
        ...

    @abstractmethod
    def data_index_from_picked_id(self, sub_id: int) -> int:
        """
        returns the index of a picked point after applying visibility filters
        
        :param sub_id: index in the visible subset
        :type sub_id: int
        :return: index in the data
        :rtype: int
        """
    @property
    def id(self) -> str:
        return self.data.id
    
    @property
    def browser_name(self) -> str:
        return self._browser_name
    
    @property
    def browser_sel_descr(self)->str | None:
        """
        Optional text descriptor of the selection (e.g. '<N> points')
        """
        return None
    
    @browser_name.setter
    def browser_name(self, browser_name: str) -> None: 
        self._browser_name = browser_name
        
      
    @property
    def renderer(self) -> Optional[vtk.vtkRenderer]:
        return self._renderer
    
    @property
    def actors(self) -> Sequence[vtk.vtkProp]:
        return self._actors
    
    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        self._apply_visibility()
        
    def on_detached(self) -> None:
        """
        Optional hook for subclasses to clean up extra state after detaching
        """
        pass

    def viewport_context_actions(self, ctx: Any, pick_result: Any) -> list[Any]:
        """
        Optional viewport right-click actions for the active layer.
        """
        return []

    def handle_viewport_left_click(self, ctx: Any, event: Any, pick_result: Any) -> bool:
        """
        Optional active-layer click handler.
        """
        return False

    # VTK Property updates
    
    def _apply_visibility(self) -> None:
        if self._renderer is None:
            return
        
        for actor in self._actors:
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(int(self.visible))
    
