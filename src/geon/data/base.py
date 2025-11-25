from abc import ABC, abstractmethod 
from typing import Optional, ClassVar, Type


class BaseData(ABC):
    """
    Base class for all data objects (e.g. point clouds, meshes, etc.).
    """
    type_id: Optional[str] = None
    
    # global counter
    _id_counters: ClassVar[dict[Type["BaseData"], int]] = {}
    
    @abstractmethod
    def save_hdf5(self) -> dict:

        ...
        
    @classmethod
    @abstractmethod
    def load_hdf5(cls, data: dict) -> "BaseData":

        ...
        
    @classmethod
    def get_type_id(cls) -> str:
        return cls.type_id or cls.__name__
    
    @classmethod    
    def get_short_type_id(cls) -> str:
        """
        short type id is the upper case portion
        """
        return ''.join([c for c in cls.get_type_id() if c.isupper()])

    @classmethod
    def _generate_id(cls) -> str:
        n = cls._id_counters.get(cls, 0) + 1
        cls._id_counters[cls] = n
        return f"{cls.get_short_type_id()}_{n:04}"
        