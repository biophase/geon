import os
import os.path as osp
from glob import glob
from pathlib import Path
import h5py

from geon.data.document import Document
from geon.version import GEON_FORMAT_VERSION
from typing import Union, Optional
from dataclasses import dataclass
from enum import Enum, auto

class DocumentState(Enum):
    UNSAVED     = auto()
    MODIFIED    = auto()
    SAVED       = auto()



@dataclass
class DocumentReference:
    _path: str
    _state: DocumentState = DocumentState.SAVED

    @classmethod
    def create(cls, path: str) -> "DocumentReference":
        return cls(path, DocumentState.MODIFIED)
    
    def load(self) -> Document:
        doc = Document.load_hdf5(self._path)
        self._state = DocumentState.SAVED
        return doc




class Dataset:
    def __init__(self, working_dir = None) -> None:
        self._working_dir : Optional[str] = working_dir
        self._doc_refs : list[DocumentReference] = []


    @property
    def working_dir(self) -> Optional[str]:
        return self._working_dir
    
    @working_dir.setter
    def working_dir(self, path: Union[Path,str]):
        path = str(path)
        self._working_dir = path

    def update_references(self):
        if self.working_dir is None:
            return
        
        self._doc_refs.clear()
        file_paths = glob(osp.join(self.working_dir, "*.hdf5"))
        file_paths.extend(glob(osp.join(self.working_dir, "*", "*.hdf5")))
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                version = f["document"].attrs["geon_format_version"]
                assert not isinstance(version,  h5py.Empty)
                if version.astype(int) > GEON_FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported GEON format version {version} in {fp}; "
                        f"current version is {GEON_FORMAT_VERSION}"
                    )
            self._doc_refs.append(DocumentReference(fp))
            
