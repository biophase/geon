from typing import Type, Dict
from geon.data.base import BaseData
from .base import BaseLayer

class LayerRegistry:
    def __init__(self):
        self._map: Dict[Type[BaseData], Type[BaseLayer]] = {}

    def register(self, data_cls: Type[BaseData], layer_cls: Type[BaseLayer]):
        if data_cls in self._map:
            raise ValueError(f"Layer already registered for {data_cls}")
        self._map[data_cls] = layer_cls

    def create_layer_for(self, data_obj: BaseData, renderer) -> BaseLayer:
        data_cls = type(data_obj)
        layer_cls = self._map[data_cls]
        return layer_cls(data_obj, renderer)

layer_registry = LayerRegistry()

def layer_for(data_cls: Type[BaseData]):
    """Decorator to register a Layer subclass for a given Data subclass."""
    def decorator(layer_cls: Type[BaseLayer]):
        layer_registry.register(data_cls, layer_cls)
        return layer_cls
    return decorator

