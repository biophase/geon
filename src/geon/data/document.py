from typing import Any
import h5py
from geon.data.base import BaseData
from geon.version import GEON_FORMAT_VERSION

class Document:
    def __init__(self):
        self.items: dict[str, BaseData]
        self.meta: dict[str, Any] = {}
        
    def save_hdf5(self, path:str):
        with h5py.File(path,'w') as f:
            f.attrs['geon_format_version'] = GEON_FORMAT_VERSION
            # TODO: add rest of save logic

    def load_hdf5(self, path:str):
        # TODO: add rest of load logic
        raise NotImplementedError