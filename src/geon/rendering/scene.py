from .base import BaseLayer
from .layer_registry import layer_registry

from geon.data.document import Document
from geon.data.base import BaseData

from geon.rendering.base import BrowserGroup

from collections import OrderedDict
from typing import Optional
import vtk

class DuplicateLayerNameError(Exception):
    pass

class Scene:
    """
    Represents the currently visible physical objects.
    Only one scene can be active at any given point
    """
    def __init__(self) -> None:
        self._layers : OrderedDict[str, BaseLayer] = OrderedDict()
        self._renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        self._doc: Document = Document()

    def add_data(self, data: BaseData) -> BaseLayer:
        layer = layer_registry.create_layer_for(data)
        if layer.id in self._layers.keys():
            raise DuplicateLayerNameError(f"Can't create duplicate layer names: {layer.id}")
        self._layers[layer.id] = layer
        layer.attach(self._renderer)
        return layer
        
        
    def get_layer(self, name:str) -> BaseLayer:
        return self._layers[name]
    
    def remove_layer(self, name:str, delete_data=True) -> None:
        layer = self._layers.pop(name)
        layer.detach()
        if delete_data:
            self._doc.remove_data(layer.id)

    def set_document(self, doc: Document) -> None:
        for layer in self._layers.values():
            layer.detach()
        self._layers.clear()
        for data in doc.scene_items.values():
            self.add_data(data)
        self._doc = doc

    def import_document(self, doc: Document) -> None:
        raise NotImplementedError
    
    @property
    def browser_report(self) -> dict:
        report = {BrowserGroup.get_human_name(group.name): dict() for group in BrowserGroup}
        
        # group items
        for k, data in self._doc.scene_items.items():

    
    @property
    def renderer(self) -> vtk.vtkRenderer:
        return self._renderer
        

    


