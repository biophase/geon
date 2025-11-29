from abc import ABC, abstractmethod
import vtk
from geon.data.base import BaseData

from typing import TypeVar, Generic, Optional, Sequence
from dataclasses import dataclass, field

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
    visible: bool
    
    # VTK
    _renderer: Optional[vtk.vtkRenderer] = field(default=None, init=False, repr=False)
    _actors: list[vtk.vtkProp] = field(default_factory= list, init=False, repr=False)
    
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
    def refresh_from_data(self) -> None:
        """
        Called when the data has changed and the layer should update its VTK pipeline
        e.g. colors, geometry ...
        
        """
        ...
        
    @property
    def renderer(self) -> Optional[vtk.vtkRenderer]:
        return self._renderer
    
    @property
    def actors(self) -> Sequence[vtk.vtkProp]:
        return self.actors
    
    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        self._apply_visibility()
        
    def on_detached(self) -> None:
        """
        Optional hook for subclasses to clean up extra state after detaching
        """
        pass
    
    # VTK Property updates
    
    def _apply_visibility(self) -> None:
        if self._renderer is None:
            return
        
        for actor in self._actors:
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(int(self.visible))
                
        
    
    