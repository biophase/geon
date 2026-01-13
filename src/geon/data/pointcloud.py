import json
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Type, Union, cast, Callable
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .base import BaseData
from .definitions import ColorMap
from .registry import register_data

from geon.util.common import decode_utf8, generate_vibrant_color
from geon.config import theme

class FieldType(Enum):
    SCALAR = auto()
    VECTOR = auto()
    COLOR  = auto()
    NORMAL = auto()
    INTENSITY   = auto()
    SEMANTIC    = auto()
    INSTANCE    = auto()
    
    @classmethod
    def get_human_name(cls, field_type: "FieldType"):

        map = {
            cls.SCALAR  : "Scalar",
            cls.VECTOR  : "Vector",
            cls.COLOR   : "Color",
            cls.NORMAL  : "Normals",
            cls.INTENSITY   : "Intensity",
            cls.SEMANTIC    : "Semantic Segmentation",
            cls.INSTANCE    : "Instance Segmentation",
            
        }
        return map[field_type]
    
    
    @property
    def human_name(self) -> str:
        return self.get_human_name(self.__getattribute__(self.name))
    
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
        self._field_added_callbacks: list[Callable[[FieldBase], None]] = []

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
                
                # FIXME: field type needs to be saved as attribute of the group
                
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
            assert data.shape[0] == self.points.shape[0],\
                f"First field shape axis {data.shape[0]} doesn't match point number {self.points.shape[0]}"
            
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
    
    # def reset_cmap_bounds(self, field_names:list[str]):
    #     for field in  self.get_fields(names=field_names):
    #         if field.color_map is not None:
    #             field.color_map.color_positions[0]

   
@dataclass
class FieldBase:
    name: str
    data: np.ndarray
    field_type: FieldType
    color_map: Optional[ColorMap] = None
    
    def save_hdf5(self, fields_group: h5py.Group) -> h5py.Group:
        field_group = fields_group.create_group(self.name)
        field_group.attrs['field_type'] = self.field_type.name
        
        field_dataset = field_group.create_dataset('data', self.data.shape,data=self.data)

        # save cmap
        if self.color_map is not None:
            self.color_map.save_h5py(field_group)
        return field_group
    
    @staticmethod
    def _read_hdf5_fieldgroup(field_group: h5py.Group) -> tuple[str, np.ndarray, FieldType, Optional[ColorMap]]:
        """
        returns (name, data, field_type, color_map)
        """
        main_dataset = field_group.get('data', None)
        if main_dataset is None or not isinstance(main_dataset, h5py.Dataset): 
            raise ValueError("Invalid format.")
        data = main_dataset[()]

        ft_attr = field_group.attrs.get("field_type")
        
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
        
        return name, data, field_type, color_map
    
    @classmethod
    def from_hdf5_fieldgroup(cls, field_group: h5py.Group) -> "FieldBase":
        name, data, field_type, color_map = cls._read_hdf5_fieldgroup(field_group)
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

    # def __hash__(self) -> int:
    #     return hash((self.id, self.name))


class SemanticSchema:
    
    def __init__(self, 
                 name:str = 'untitled_schema',
                 semantic_classes : List[SemanticClass] = [
                     SemanticClass(
                         -1, 
                         '_unlabeled', 
                         theme.DEFAULT_SEGMENTATION_COLOR)
                     ]):
        self.name = name
        self.semantic_classes = semantic_classes

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
    def from_hdf5_fieldgroup(cls, field_group: "h5py.Group"):
        dataset = field_group.get("semantic_schema")
        assert  isinstance(dataset, h5py.Dataset), "Invalid file."
        val = dataset[()]
        if isinstance(val, (bytes, bytearray)):
            s = val.decode("utf-8")
        else:
            s = str(val)
        schema = cls.from_dict(json.loads(s))
        name_attr = dataset.attrs.get("name", None)
        if name_attr is not None:
            if isinstance(name_attr, (bytes, bytearray)):
                schema.name = name_attr.decode("utf-8")
            else:
                schema.name = str(name_attr)
        return schema
        

    def signature(self) -> tuple:
        classes = sorted(self.semantic_classes, key =lambda c: c.id)
        return tuple((c.id, c.name, tuple(c.color)) for c in classes)
        
    # def __hash__(self) -> int:
    #     return hash(tuple(self.semantic_classes))
    
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
        seg_data = np.asarray(seg_data, np.int32).reshape(-1)

        ids = [s.id for s in self.semantic_classes]
        max_id = max(max(ids), seg_data.max())
        map_arr = np.full((max_id + 2, 3), theme.DEFAULT_SEGMENTATION_COLOR, np.uint8)

        for cls in self.semantic_classes:
            if cls.id >= 0:
                map_arr[cls.id] = np.array(cls.color, dtype=np.uint8)
            elif cls.id == -1:
                map_arr[-1] = np.array(cls.color, dtype=np.uint8)

        seg_clipped = np.clip(seg_data, -1, map_arr.shape[0] - 1)
        return map_arr[seg_clipped]

        
    @classmethod
    def scan_h5(cls, path: Union[str, Path]) -> Dict[str, "SemanticSchema"]:
        """
        Fast scan of a GEON .h5/.hdf5 document file to extract SemanticSchema objects.

        Key format:
            "{document_name}/{pointcloud_id}/{semantic_field_name}/{schema_name}"

        Notes:
        - Does NOT load point cloud points or segmentation data arrays.
        - Reads only small attributes and the scalar-string 'semantic_schema' dataset.
        """
        
        path = Path(path)
        out: Dict[str, SemanticSchema] = {}
        
        def _decode_utf8(value) -> str:
            """Local fallback; replace with your decode_utf8 if you prefer."""
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8")
            return str(value)

        with h5py.File(path, "r") as f:
            doc_grp = f.get("document")
            if not isinstance(doc_grp, h5py.Group):
                raise ValueError(f"No '/document' group in file: {path}")

            # document name (stored as attribute on /document)
            doc_name_attr = doc_grp.attrs.get("name", "UnnamedDocument")
            document_name = _decode_utf8(doc_name_attr)

            # Iterate document children (e.g. PCD_0001, ...)
            for pc_id in doc_grp.keys():
                pc_grp = doc_grp.get(pc_id)
                if not isinstance(pc_grp, h5py.Group):
                    continue

                # Optional: ensure it's a PointCloudData group (fast attribute check)
                type_id = pc_grp.attrs.get("type_id", pc_grp.attrs.get("type"))
                if type_id is not None:
                    type_id_str = _decode_utf8(type_id)
                    # your PointCloudData.save_hdf5 uses type_id = self.get_type_id()
                    # which is likely "PointCloudData"
                    if type_id_str not in ("PointCloudData", "PointCloud"):  # keep lenient if you rename later
                        continue

                fields_grp = pc_grp.get("fields")
                if not isinstance(fields_grp, h5py.Group):
                    continue

                # Iterate fields under /fields (e.g. "intensity", "semantics", ...)
                for field_name in fields_grp.keys():
                    field_grp = fields_grp.get(field_name)
                    if not isinstance(field_grp, h5py.Group):
                        continue
                    
                    # if 'field_type' not in field_grp.attrs:
                    #     continue
                    ft = field_grp.attrs.get('field_type')
                    # Fast check: field_type is stored on the *data dataset* attrs in your save code
                    # data_ds = field_grp.get("data")
                    # if not isinstance(data_ds, h5py.Dataset):
                    #     continue
                    


                    # ft = data_ds.attrs.get("field_type")
                    if ft is None:
                        continue
                    field_type = _decode_utf8(ft)
                    if field_type != "SEMANTIC":
                        continue

                    # The schema is stored as a scalar-string dataset "semantic_schema" with JSON payload
                    schema_ds = field_grp.get("semantic_schema")
                    if not isinstance(schema_ds, h5py.Dataset):
                        # semantic field without schema dataset: skip (or create default if you prefer)
                        continue

                    raw = schema_ds[()]  # scalar string/bytes, tiny
                    schema_json = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    semantic_dict = json.loads(schema_json)

                    schema = SemanticSchema.from_dict(semantic_dict)

                    schema_name_attr = schema_ds.attrs.get("name", schema.name)
                    schema_name = _decode_utf8(schema_name_attr)
                    schema.name = schema_name  # keep name consistent with stored attribute

                    # Build key: document_name/pointcloud_name/semantic_field_name/semantic_schema_name
                    key = f"{document_name}/{pc_id}/{field_name}/{schema_name}"
                    out[key] = schema

        return out

    
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
        data = self.data.reshape(-1)
        hashed = (data * 2654435761) & 0xFFFFFFFF
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
            data = np.asarray(data, np.int32).reshape(-1)
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
        # dataset = field_group.get('data')
        # assert isinstance(dataset, h5py.Dataset), "Invalid file."
        # data = dataset[()]
        # name = field_group.name
        # assert isinstance(name, str)
        # name = name.split("/")[-1]
        
        name, data, _, _ = cls._read_hdf5_fieldgroup(field_group)

        schema = SemanticSchema.from_hdf5_fieldgroup(field_group)

        return cls(name=name, data=data, schema=schema)
    
  

    def get_next_id(self) -> int:
        return mex(self.data)
         
    # def replace_semantic_schema (self, new_schema : SemanticSchema, by : Literal['id','name'] = 'id') -> None:
    #     raise NotImplementedError
    
    def remap(
        self,
        old_2_new_ids: list[tuple[int,int]],
        ):
        """
        Remap the field data.
        :param old_to_new: pairs of indices [(old id, new id), ...]
        :type old_to_new: list[tuple[int, int]]
        """

        id_min = int(self.data.min())
        id_max = int(self.data.max())

        # construct mapping[old_idx - id_min] -> new_idx
        mapping = np.arange(id_min, id_max + 1, dtype=np.int32)
        for old_id, new_id in old_2_new_ids:
            if old_id < id_min or old_id > id_max:
                continue
            mapping[old_id - id_min] = new_id
        self.data = mapping[self.data - id_min]
    
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
    
