from typing import Type, Dict, Any, TypeVar, Callable
from geon.data.base import BaseData
from .base import BaseLayer

class LayerRegistry:
    def __init__(self):
        self._map: Dict[Type[BaseData], Type[BaseLayer]] = {}

    def register(self, data_cls: Type[BaseData], layer_cls: Type[BaseLayer]):
        if data_cls in self._map:
            raise ValueError(f"Layer already registered for {data_cls}")
        self._map[data_cls] = layer_cls

    def create_layer_for(self, data_obj: BaseData) -> BaseLayer:
        data_cls = type(data_obj)
        layer_cls = self._map[data_cls]
        return layer_cls(data_obj)

LAYER_REGISTRY = LayerRegistry()

DataT = TypeVar("DataT", bound=BaseData)
LayerT = TypeVar("LayerT", bound=BaseLayer[Any])

def layer_for(data_cls: type[DataT]) -> Callable[[type[LayerT]], type[LayerT]]:
    """Decorator to register a Layer subclass for a given Data subclass."""
    def decorator(layer_cls: type[LayerT]) -> type[LayerT]:
        LAYER_REGISTRY.register(data_cls, layer_cls)
        return layer_cls
    return decorator

