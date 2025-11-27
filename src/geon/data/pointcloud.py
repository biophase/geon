import numpy as np
from .registry import register_data
from .base import BaseData


from typing import Tuple, TypedDict, List, Dict, Optional, Literal
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from enum import Enum, auto
import pickle

class FieldType(Enum):
    SCALAR = auto()
    VECTOR = auto()
    COLOR  = auto()
    INTENSITY   = auto()
    SEMANTIC    = auto()
    INSTANCE    = auto()

@register_data
class PointCloudData(BaseData):
    def __init__(self, points: np.ndarray):
        super().__init__()
        self.points = points
        self._fields : list[FieldBase] = []

    def save_hdf5(self) -> Dict:
        raise NotImplementedError
        return super().save_hdf5()
    
    @classmethod
    def load_hdf5(cls, data:dict):
        raise NotImplementedError
    
    @property
    def field_names(self)->list[str]:
        return [f.name for f in self._fields]
    
    def add_field(self, 
                  name:Optional[str]=None, 
                  data:Optional[np.ndarray]=None, 
                  field_type:Optional[FieldType]=None,
                  vector_dim_hint: int = 1,
                  default_fill_value:float = 0.,
                  dtype_hint = np.float32
                  ) -> None:
        
        assert name not in self.field_names, \
            "Field names should not be duplicates in same point cloud."
        
        if field_type is None:
            if vector_dim_hint == 1:
                field_type = FieldType.SCALAR
            else:
                field_type = FieldType.VECTOR

        if name is None:
            field_prefix = 'Field_'
            taken_ids = [int(n.replace(field_prefix,'')) for n in self.field_names]
            new_id = mex(np.array(taken_ids))
            name = f"{field_prefix}{new_id:04}"
        if data is not None:
            assert data.ndim == 2, \
                f"Fields should have two dims but got: {data.shape}"
            assert data.shape[0] == self.points.shape[0]
            field = FieldBase(name, data, field_type)

        else:
            
            shape = (self.points.shape[0], vector_dim_hint)
            data = np.full(shape, default_fill_value, dtype_hint)
            field = FieldBase(name, data, field_type)

        self._fields.append(field)

    def remove_field(self,
                     name: Optional[str] = None,
                     field_type: Optional[FieldType] = None,
                     ):
        if name is not None:

    def get_field(self,
            name: Optional[str | list[str]] = None,
            field_type: Optional[FieldType] = None
            )->list["FieldBase"]:
        return []
        

   
@dataclass
class FieldBase:
    name: str
    data: np.ndarray
    field_type: FieldType

    


class SemanticDescription(TypedDict):
    """
    helper type
    """
    name: str
    color:Tuple[float,float,float]


@dataclass
class SemanticClass:
    """
    e.g. name='column', id=0, color=(1.,0.,0.)
    """

    id:     int
    name:   str
    color:  Tuple[float,float,float]

    def __hash__(self) -> int:
        return hash((self.id, self.name))


class SemanticSchema:
    
    def __init__(self):
        self.semantic_classes : List[SemanticClass] = [SemanticClass(-1, '_unlabeled', (0.8,0.8,0.8))]

    def to_dict(self) -> Dict[str, dict]:
        return {
            str(s.id): {'name': s.name, 'color': s.color}
            for s in self.semantic_classes
        }
    
    @classmethod
    def from_json(cls, json_path:str) -> "SemanticSchema":
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, json_path: str) -> None:
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, semantic_dict : Dict[str, SemanticDescription]) -> "SemanticSchema":
        instance = cls.__new__(cls)
        instance.semantic_classes = []
        for key, value in semantic_dict.items():
            class_id = int(key)
            r, g, b = value["color"]
            semantic_class = SemanticClass(class_id, value["name"], (r,g,b))
            instance.semantic_classes.append(semantic_class)
        instance.semantic_classes.sort(key=lambda s: s.id)
        return instance
        
    
    def __hash__(self) -> int:
        return hash(tuple(self.semantic_classes))
    
    def add_semantic_class(self, semantic_class : SemanticClass) -> None:
        assert semantic_class.id not in [s.id for s in self.semantic_classes], \
            f"Index {semantic_class.id} already exists in schema."
        assert semantic_class.name not in [s.name for s in self.semantic_classes], \
            f"Name {semantic_class.name} already exists in schema."
        self.semantic_classes.append(semantic_class)
        self.semantic_classes.sort(key = lambda x: x.id)

    def remove_semantic_class(self, id:int) -> None:
        assert id in [s.id for s in self.semantic_classes], f"Index {id} is does not exist and can't be deleted."
        self.semantic_classes = [s for s in self.semantic_classes if s.id != id]

    def reindex(self) -> Dict[int, int]:
        self.semantic_classes.sort(key=lambda x: x.id)
        id_map: Dict[int, int] = {}
        for i, s in enumerate(self.semantic_classes):
            old_id = s.id
            if old_id == -1:
                id_map[old_id] = old_id
                continue
            s.id = i
            id_map[old_id] = i
        return id_map



    
def mex(data:np.ndarray)->int:
    """
    returns the minimum excluded value
    """
    if data.ndim == 0: 
        return 0
    else:
        assert data.ndim == 1, \
            f"Can only compute MEX on flat arrays but got {data.shape}"

    nonneg = data[data >= 0]
    size = len(data) + 1
    present = np.zeros(size, dtype=bool)
    present[nonneg[nonneg < size]] = True
    return np.flatnonzero(~present)[0]
    


class InstanceSegmentation(FieldBase):
    def __init__(self, name:str, data: Optional[np.ndarray], size: Optional[int]):
        if data is not None:
            assert isinstance (data, np.ndarray), f"data should be a numpy array but got {type(data)}"
            super().__init__(name, data.astype(np.int32), FieldType.INSTANCE)
        elif size is not None:
            super().__init__(name, np.zeros(size, dtype=np.int32), FieldType.INSTANCE)
        
        else:
            raise ValueError("Either size or data should be provided.")
        
        
    def get_next_instance_id(self) -> int:
        return mex(self.data)
        
    

class SemanticSegmentation(FieldBase):
    def __init__(self,  name:str, 
                 data:  Optional[np.ndarray], 
                 size:  Optional[int],
                 schema:SemanticSchema
                 ):
        if data is not None:
            assert isinstance (data, np.ndarray), f"data should be a numpy array but got {type(data)}"
            super().__init__(name, data.astype(np.int32), FieldType.SEMANTIC)
        elif size is not None:
            super().__init__(name, np.full(size,-1, dtype=np.int32), FieldType.SEMANTIC)
        
        else:
            raise ValueError("Either size or data should be provided.")
        
        self.schema = schema
        
    def get_next_id(self) -> int:
        return mex(self.data)
        

    
    def replace_semantic_schema (self, new_schema : SemanticSchema, by : Literal['id','name'] = 'id') -> None:
        raise NotImplementedError
    
    def reindex_semantic(self):
        raise NotImplementedError # TODO: should reindex the containers, not only the schema. requires mapping output from schema method
    
    def save(self, file_path: str) -> None:
        raise DeprecationWarning
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path) -> "SemanticSegmentation":
        raise DeprecationWarning
        with open(file_path, "rb") as f:
            seg = pickle.load(f)
        return seg
    
    
        
    

        


