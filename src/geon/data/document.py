from typing import Any, Union
from enum import Enum, auto
import h5py
from geon.data.base import BaseData
from geon.version import GEON_FORMAT_VERSION
from geon.data.registry import data_registry
from pathlib import Path
import json
import os.path as osp
import os
import numpy as np


from typing import Optional




class Document:
    def __init__(self, name:str = 'Untitled'):
        self.name = name
        self.scene_items: dict[str, BaseData] = {}
        self.telemetry: list[str] = []
        self.meta: dict[str, Any] = {
            'name':name
            }
        
        

    def save_hdf5(self, path : Union[str, Path]):
        dir_path, file_path = osp.split(path)
        os.makedirs(dir_path, exist_ok=True)
        path = Path(path)      
        with h5py.File(path,'w') as f:
            root = f.create_group("document")
            self._save_to_group(root)

    def _save_to_group(self, group: h5py.Group) -> None:
        # doc metadata
        group.attrs['geon_format_version'] = GEON_FORMAT_VERSION
        group.attrs['type'] = "Document"
        for k, v in self.meta.items():
            group.attrs[k] = v

        # telemetry
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            "telemetry",
            data=np.asarray(self.telemetry, dtype=dtype),
            dtype=dtype,
            shape=(len(self.telemetry),),
        )
        
        # doc children
        
        for k, v in self.scene_items.items():
            child_group = group.create_group(v.id)
            v.save_hdf5(child_group)
            
            
    def __repr__(self):
        return ''.join(["Document containing:\n"] + [f"- '{n}'\n" for n in self.scene_items.keys()])
        
    def add_data(self, data: BaseData):
        assert data.id not in self.scene_items.keys(),\
            f"Tried adding duplicate item id {data.id}"
        self.scene_items[data.id] = data
    
    def remove_data(self,id:str)->None:
        data = self.scene_items.pop(id)
        
    @classmethod
    def load_hdf5(cls, path:Union[str, Path]):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        
        def _decode(value: Any) -> Any:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        with h5py.File(path, 'r') as f:
            if "document" not in f:
                raise ValueError(f"No document group found in {path}")
            group = f["document"]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"Expected 'document' to be an HDF5 group in {path}, got {type(group).__name__}")
            doc_type = _decode(group.attrs.get("type"))
            if doc_type != "Document":
                raise ValueError(f"Invalid root type '{doc_type}' in {path}")
            version = group.attrs.get("geon_format_version")
            if version != GEON_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported GEON format version {version}, expected {GEON_FORMAT_VERSION}"
                )
            
            loaded_meta: dict[str, Any] = {}
            for key, value in group.attrs.items():
                if key in {"geon_format_version", "type"}:
                    continue
                loaded_meta[key] = _decode(value)

            telemetry_ds = group.get("telemetry")
            telemetry_entries: list[str] = []
            if isinstance(telemetry_ds, h5py.Dataset):
                raw = telemetry_ds[()]
                flat = raw.ravel() if hasattr(raw, "ravel") else raw
                for entry in flat:
                    if isinstance(entry, bytes):
                        telemetry_entries.append(entry.decode("utf-8"))
                    else:
                        telemetry_entries.append(str(entry))
            
            loaded_items: dict[str, BaseData] = {}
            for child_name in group.keys():
                if child_name == "telemetry":
                    continue
                child_node = group[child_name]
                if not isinstance(child_node, h5py.Group):
                    raise ValueError(f"Expected HDF5 group for item '{child_name}', got {type(child_node).__name__}")
                child_group = child_node
                type_attr = child_group.attrs.get("type_id", child_group.attrs.get("type"))
                if type_attr is None:
                    raise ValueError(f"Missing type information for item '{child_name}'")
                type_id = _decode(type_attr)
                data_cls = data_registry.get(str(type_id))
                data_obj = data_cls.load_hdf5(child_group)
                
                # Ensure ids stay in sync with the group name
                if getattr(data_obj, "id", None) != child_name:
                    data_obj.id = child_name
                loaded_items[child_name] = data_obj
        
        doc = cls()
        doc.meta = loaded_meta
        doc.telemetry = telemetry_entries
        if "name" in loaded_meta:
            doc.name = str(loaded_meta["name"])
        doc.scene_items = loaded_items
        return doc
    
    
    
    
    
    


    
