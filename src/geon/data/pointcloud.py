import json
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import h5py
import numpy as np
from numpy.typing import NDArray

from .base import BaseData
from .registry import register_data

class FieldType(Enum):
    SCALAR = auto()
    VECTOR = auto()
    COLOR  = auto()
    INTENSITY   = auto()
    SEMANTIC    = auto()
    INSTANCE    = auto()
    
    @classmethod
    def get_human_name(cls, field_type: "FieldType"):

        map = {
            cls.SCALAR  : "Scalar",
            cls.VECTOR  : "Vector",
            cls.COLOR   : "Color",
            cls.INTENSITY   : "Intensity",
            cls.SEMANTIC    : "Semantic Segmentation",
            cls.INSTANCE    : "Instance Segmentation",
        }
        return map[field_type]
        

@register_data
class PointCloudData(BaseData):
    """
    Docstring for PointCloudData
    """
    def __init__(self, points: np.ndarray):
        super().__init__()
        self.points = points
        self._fields : list[FieldBase] = []

    def save_hdf5(self, group: h5py.Group) -> Dict:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id

        group.create_dataset("points", data=self.points)

        if self._fields:
            fields_group = group.create_group("fields")
            for field in self._fields:
                dataset = fields_group.create_dataset(field.name, data=field.data)
                dataset.attrs["field_type"] = field.field_type.name

        return {"num_points": self.points.shape[0], "num_fields": len(self._fields)}
    
    @classmethod
    def load_hdf5(cls, group: h5py.Group):
        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        dataset = group.get("points")
        if dataset is None or not isinstance(dataset, h5py.Dataset):
            raise ValueError("HDF5 group for PointCloudData must contain a 'points' dataset.")

        points = dataset[()]
        obj = cls(points)

        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = _decode(stored_id)

        fields_group = group.get("fields")
        if isinstance(fields_group, h5py.Group):
            for name, dataset in fields_group.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                data = dataset[()]
                ft_attr = dataset.attrs.get("field_type")
                field_type = FieldType[_decode(ft_attr)] if ft_attr is not None else FieldType.SCALAR
                obj._fields.append(FieldBase(name, data, field_type))
        return obj
    
    @property
    def field_names(self)->list[str]:
        return [f.name for f in self._fields]
    
    def add_field(self, 
                  name:Optional[str]=None, 
                  data:Optional[np.ndarray]=None, 
                  field_type:Optional[FieldType]=None,
                  vector_dim_hint: int = 1,
                  default_fill_value:float = 0.,
                  dtype_hint = np.float32,
                  schema: Optional["SemanticSchema"] = None
                  ) -> None:
        
        assert name not in self.field_names, \
            "Field names should not be duplicates in same point cloud."
        assert name != 'points',\
            "Field name 'points' is reserved."
        
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
            
        # fields with specialized classes
        if field_type == FieldType.SEMANTIC:
            if data is not None:
                field = SemanticSegmentation(name, data, schema=schema)
            else:
                field = SemanticSegmentation(name, size=self.points.shape[0], schema=schema)
            
        elif field_type == FieldType.INSTANCE:
            if data is not None:
                field = InstanceSegmentation(name, data)
            else:
                field = InstanceSegmentation(name, size=self.points.shape[0])
        
        # generic fields
        else:
            if data is not None:
                field = FieldBase(name, data, field_type)
            else:
                shape = (self.points.shape[0], vector_dim_hint)
                data = np.full(shape, default_fill_value, dtype_hint)
                field = FieldBase(name, data, field_type)

        self._fields.append(field)

    def remove_fields(self,
                     names: Optional[str | list[str]] = None,
                     field_type: Optional[FieldType] = None,
                     ):

        if names is None and field_type is None:
            raise ValueError("Either a name or field type should be supplied to the query")

        name_set: Optional[set[str]] = None
        if names is not None:
            if isinstance(names, (list, tuple, set)):
                name_set = set(names)
            else:
                name_set = {names}

        def should_remove(field: FieldBase) -> bool:
            if name_set is not None and field.name not in name_set:
                return False
            if field_type is not None and field.field_type != field_type:
                return False
            return True

        self._fields = [field for field in self._fields if not should_remove(field)]
        
    def get_fields(self,
            names: Optional[str | list[str]] = None,
            field_type: Optional[FieldType] = None
            )->list["FieldBase"]:
        if names is not None:
            names = names if isinstance(names, (list, tuple)) else [names]
            if field_type is not None:
                return [f for f in self._fields if f.name in names and f.field_type == field_type]
            return [f for f in self._fields if f.name in names]
        else:
            if field_type is not None:
                return [f for f in self._fields if f.field_type == field_type]
            return self._fields
    
    def __getitem__(self, name:str) -> np.ndarray:
        if name == 'points':
            return self.points
        return self.get_fields(names=name)[0].data
    
    @property
    def colors(self) -> Optional["FieldBase"]:
        fields = self.get_fields(field_type=FieldType.COLOR)
        if len(fields):
            return fields[0]
        else:
            return None
    @property
    def intensity(self) -> Optional["FieldBase"]:
        fields = self.get_fields(field_type=FieldType.INTENSITY)
        if len(fields):
            return fields[0]
        else:
            return None
        
    def to_structured_array(self) -> np.ndarray:
        num_points = self.points.shape[0]
        coord_names = ('x', 'y', 'z')
        dtype_fields: list[tuple] = []
        assignments: list[tuple[str, np.ndarray]] = []

        for idx in range(self.points.shape[1]):
            field_name = coord_names[idx] if idx < len(coord_names) else f"coord_{idx}"
            dtype_fields.append((field_name, self.points.dtype))
            assignments.append((field_name, self.points[:, idx]))

        for field in self._fields:
            data = field.data
            if data.shape[0] != num_points:
                raise ValueError(f"Field '{field.name}' has {data.shape[0]} entries, expected {num_points}.")

            if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
                values = data if data.ndim == 1 else data[:, 0]
                dtype_fields.append((field.name, values.dtype))
                assignments.append((field.name, values))
            elif data.ndim == 2:
                shape = (data.shape[1],)
                dtype_fields.append((field.name, data.dtype, shape))
                assignments.append((field.name, data))
            else:
                raise ValueError(f"Unsupported field dimensionality for '{field.name}': {data.shape}.")

        structured = np.empty(num_points, dtype=dtype_fields)
        for name, values in assignments:
            structured[name] = values

        return structured

   
@dataclass
class FieldBase:
    name: str
    data: np.ndarray
    field_type: FieldType
    
    def save_hdf5(self, group: h5py.Group) -> None:
        dataset = group.create_dataset(self.name, self.data)
        dataset.attrs['field_type'] = self.field_type.name
        

    


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
    def __init__(self, name:str, 
                 data: Optional[np.ndarray] = None, 
                 size: Optional[int] = None
                 ):
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
                 data:  Optional[np.ndarray]=None,
                 size:  Optional[int]=None,
                 schema:Optional[SemanticSchema]=None,
                 ):
        if data is not None:
            assert isinstance (data, np.ndarray), f"data should be a numpy array but got {type(data)}"
            super().__init__(name, data.astype(np.int32), FieldType.SEMANTIC)
        elif size is not None:
            super().__init__(name, np.full(size,-1, dtype=np.int32), FieldType.SEMANTIC)
        
        else:
            raise ValueError("Either size or data should be provided.")
        
        if schema is None:
            schema = SemanticSchema()
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
    