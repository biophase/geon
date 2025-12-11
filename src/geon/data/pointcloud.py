import json
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Type, Union, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from .base import BaseData
from .definitions import ColorMap
from .registry import register_data

from geon.utils.common import decode_utf8, generate_vibrant_color
from config import theme

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
    
    @classmethod
    def get_class (cls, field_type:"FieldType") -> Type["FieldBase"]:
        """
        Mapping in reconstrucing specialized field types from the enum e.g. when reading h5py files
        """
        return {
            cls.SEMANTIC : SemanticSegmentation,
            cls.INSTANCE : InstanceSegmentation
        }.get(field_type, FieldBase)
        

@register_data
class PointCloudData(BaseData):
    """
    Docstring for PointCloudData
    """
    def __init__(self, points: np.ndarray):
        super().__init__()
        self.points = points
        self._fields : list[FieldBase] = []

    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        group.attrs["type_id"] = self.get_type_id()
        group.attrs["id"] = self.id

        group.create_dataset("points", data=self.points)

        if self._fields:
            fields_group = group.create_group("fields")
            for field in self._fields:
                field.save_hdf5(fields_group)
                

        return group
    
    @classmethod
    def load_hdf5(cls, group: h5py.Group):

        field_group = group.get("points")
        if field_group is None or not isinstance(field_group, h5py.Dataset):
            raise ValueError("HDF5 group for PointCloudData must contain a 'points' dataset.")

        points = field_group[()]
        obj = cls(points)

        stored_id = group.attrs.get("id")
        if stored_id is not None:
            obj.id = decode_utf8(stored_id)

        fields_group = group.get("fields")
        if isinstance(fields_group, h5py.Group):
            for name, field_group in fields_group.items():
                if not isinstance(field_group, h5py.Group):
                    continue

                ft_attr = field_group.attrs.get("field_type")
                field_type = FieldType[decode_utf8(ft_attr)] if ft_attr is not None else FieldType.SCALAR
                field_class = FieldType.get_class(field_type)
                field = field_class.from_hdf5_fieldgroup(field_group)
                obj._fields.append(field)

        return obj
    
    @property
    def field_names(self)->list[str]:
        return [f.name for f in self._fields]
    
    @property
    def field_num(self) -> int:
        return len(self.field_names)
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
            taken_ids: list[int] = []
            for n in self.field_names:
                if n.startswith(field_prefix):
                    suffix = n[len(field_prefix):]
                    try:
                        taken_ids.append(int(suffix))
                    except ValueError:
                        # ignore non-numeric suffixes
                        pass
                
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
            field_type: Optional[FieldType] = None,
            field_index: Optional[int] = None
            )->list["FieldBase"]:
        if names is not None:
            names = names if isinstance(names, (list, tuple)) else [names]
            if field_type is not None:
                return [f for f in self._fields if f.name in names and f.field_type == field_type]
            return [f for f in self._fields if f.name in names]
        elif field_index is not None:
            return [self._fields[field_index]]
        else:
            if field_type is not None:
                return [f for f in self._fields if f.field_type == field_type]
            return [f for f in self._fields]
        
    
    def __getitem__(self, name: str) -> np.ndarray:
        if name == "points":
            return self.points
        fields = self.get_fields(names=name)
        if not fields:
            raise KeyError(f"Field '{name}' not found")
        return fields[0].data

    
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
    color_map: Optional[ColorMap] = None
    
    def save_hdf5(self, fields_group: h5py.Group) -> h5py.Group:
        field_group = fields_group.create_group(self.name)
        field_dataset = field_group.create_dataset('data', self.data.shape,data=self.data)
        field_dataset.attrs['field_type'] = self.field_type.name

        # save cmap
        if self.color_map is not None:
            self.color_map.save_h5py(field_group)
        return field_group
    
    @classmethod
    def from_hdf5_fieldgroup(cls, field_group: h5py.Group) -> "FieldBase":
        main_dataset = field_group.get('data', None)
        if main_dataset is None or not isinstance(main_dataset, h5py.Dataset): 
            raise ValueError("Invalid format.")
        data = main_dataset[()]

        ft_attr = main_dataset.attrs.get("field_type")
        field_type = FieldType[decode_utf8(ft_attr)] if ft_attr is not None else FieldType.SCALAR
        name = field_group.name
        assert isinstance(name, str)
        name = name.split("/")[-1]

        color_map_ds = field_group.get('color_map')
        if color_map_ds is not None:
            assert isinstance(color_map_ds, h5py.Dataset)
            color_map = ColorMap.load_h5py(color_map_ds)
        else:
            color_map = None
        
        return cls(name, data, field_type, color_map)
    


class SemanticDescription(TypedDict):
    """
    helper type
    """
    name: str
    color:Tuple[int,int,int]


@dataclass
class SemanticClass:
    """
    e.g. name='column', id=0, color=(255,0,128)
    """

    id:     int
    name:   str
    color:  Tuple[int,int,int]

    def __hash__(self) -> int:
        return hash((self.id, self.name))


class SemanticSchema:
    
    def __init__(self, name:str = 'untitled_schema'):
        self.name = name
        self.semantic_classes : List[SemanticClass] = [SemanticClass(-1, '_unlabeled', theme.DEFAULT_SEGMENTATION_COLOR)]

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
        schema = cls()
        schema.semantic_classes = []
        for key, value in semantic_dict.items():
            class_id = int(key)
            r, g, b = value["color"]
            semantic_class = SemanticClass(class_id, value["name"], (r,g,b))
            schema.semantic_classes.append(semantic_class)
        schema.semantic_classes.sort(key=lambda s: s.id)
        return schema
        
    def save_h5py (self, field_group: h5py.Group) -> None:
        dt = h5py.string_dtype(encoding="utf-8")
        ds = field_group.create_dataset("semantic_schema", data=np.array(json.dumps(self.to_dict()), dtype=dt), dtype=dt)
        ds.attrs['name'] = self.name
    
    @classmethod
    def from_h5py_fieldgroup(cls, field_group: "h5py.Group"):
        dataset = field_group.get("semantic_schema")
        assert  isinstance(dataset, h5py.Dataset), "Invalid file."
        val = dataset[()]
        if isinstance(val, (bytes, bytearray)):
            s = val.decode("utf-8")
        else:
            s = str(val)
        return cls.from_dict(json.loads(s))
        

    def __hash__(self) -> int:
        return hash(tuple(self.semantic_classes))
    
    def add_semantic_class(self, semantic_class : Union[str, SemanticClass]) -> None:
        if isinstance (semantic_class,str):
            semantic_class = SemanticClass(self.get_next_id(),semantic_class, generate_vibrant_color())
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
    def get_next_id(self) -> int:
        return mex(np.array([c.id for c in self.semantic_classes]))
    
    def by_id (self, id:int) -> SemanticClass:
        """return the first class that has an id"""
        result = [s for s in self.semantic_classes if s.id==id]
        if len(result):
            return result[0]
        else: 
            raise IndexError(f"Index {id} is not in the schema")

    def get_color_array(self, seg_data: NDArray[np.int32]) -> NDArray[np.uint8]:
        seg_data = np.asarray(seg_data, np.int32)
        
        ids = [s.id for s in self.semantic_classes]
        size = max(max(ids), seg_data.max())

        # populate the mapping array. -1 maps at end
        map_arr = np.zeros((size + 2, 3), np.uint8)
        map_arr[:] = theme.DEFAULT_SEGMENTATION_COLOR
        for i in range(size):
            map_arr[i] = np.array(self.by_id(i).color)
       
        return map_arr[seg_data]
        

    
def mex(data:np.ndarray)->int:
    """
    returns the minimum excluded value
    """
    if data.ndim == 0: 
        return 0
    elif len(data) == 0:
        return 0
    else:
        assert data.ndim == 1, \
            f"Can only compute MEX on flat arrays but got {data.shape}"
    assert np.issubdtype(data.dtype, np.integer)
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
        
    def get_color_array(self) -> NDArray[np.uint8]:
        """
        returns colors in range 0..255
        """
        
        # Knuth multiplicative hash
        hashed = (self.data * 2654435761) & 0xFFFFFFFF
        h = (hashed.astype(np.float32) / np.float32(2**32)) 
        s = np.full_like(h, .9, dtype=np.float32)
        v = np.full_like(h, .9, dtype=np.float32)
        hsv = np.stack([h,s,v], axis=1)
        import matplotlib.colors as mcolors
        rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
        return rgb
        
    
    def get_next_instance_id(self) -> int:
        return mex(self.data)
    
    # @classmethod
    # def from_hdf5_fieldgroup(cls, dataset: h5py.Dataset) -> "InstanceSegmentation":
    #     data = dataset[()]
    #     name = dataset.name
    #     assert isinstance(name, str)            
    #     name = name.split("/")[-1]
    #     return cls(name=name, data=data)
        
    

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
        
    def save_hdf5(self, fields_group: h5py.Group) -> h5py.Group:
        field_group = super().save_hdf5(fields_group)
        self.schema.save_h5py(field_group=field_group)
        

        return field_group      
    
    @classmethod
    def from_hdf5_fieldgroup(cls, field_group: h5py.Group) -> "SemanticSegmentation":
        dataset = field_group.get('data')
        assert isinstance(dataset, h5py.Dataset), "Invalid file."
        data = dataset[()]
        name = dataset.name
        assert isinstance(name, str)
        name = name.split("/")[-1]
        

        schema = SemanticSchema.from_h5py_fieldgroup(field_group)

        return cls(name=name, data=data, schema=schema)
    
  

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
    
