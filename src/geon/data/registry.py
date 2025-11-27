from typing import Type, TypeVar
from .base import BaseData

class DataRegistry:
    def __init__(self):
        self._by_type_id: dict[str, Type[BaseData]] = {}
        
    def register(self, cls:Type[BaseData]):
        if cls.get_type_id() in self._by_type_id:
            raise ValueError(f"Duplicate type_id: {cls.type_id}")
        self._by_type_id[cls.get_type_id()] = cls
        
    def get(self, type_id:str) -> Type[BaseData]:
        return self._by_type_id[type_id]
    

# type checker fix
T = TypeVar("T", bound=BaseData)
    
data_registry = DataRegistry()

def register_data(cls: Type[T]) -> Type[T]:
    data_registry.register(cls)
    return cls
    