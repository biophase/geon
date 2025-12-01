from abc import ABC, abstractmethod 
from typing import Optional, ClassVar, Type, TypeVar
import h5py


class BaseData(ABC):
    """
    Base class for all top-level document data objects (e.g. point clouds, meshes, etc.).
    """
    type_id: Optional[str] = None
    
    # global counter
    _id_counters: ClassVar[dict[Type["BaseData"], int]] = {}
    
    @abstractmethod
    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        """
        Save this object into the given HDF5 group.

        The method should:
          - create datasets/groups inside `group`
          - set any relevant attributes on `group`
        It MUST NOT close the file or the group.
        """

        ...
        
    @classmethod
    @abstractmethod
    def load_hdf5(cls, group: h5py.Group) -> "BaseData":
        """
        Load this object from an HDF5 group and return an instance.
        The method should read attributes, datasets and subgroups
        inside `group`, build the corresponding in-memory object,
        and return it.
        """
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
    
    def __init__(self):
        self.id = self._generate_id()
        