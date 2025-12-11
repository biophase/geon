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
    _name: str
    _path: Optional[str]
    _state: DocumentState = DocumentState.SAVED

    # @classmethod
    # def create(cls, path: str) -> "DocumentReference":
    #     return cls(path, DocumentState.MODIFIED)
    
    def load(self) -> Document:
        if self.path is None:
            raise Exception("Attempted to load a reference with no path")
        doc = Document.load_hdf5(self.path)
        self._state = DocumentState.SAVED
        return doc
    
    @property
    def path(self) -> Optional[str]:
        return self._path
    
    @path.setter
    def path(self, path:str)->None:
        self._path = path
        
    @property
    def state(self) -> DocumentState:
        return self._state
    
    @state.setter
    def state(self, state: DocumentState) -> None:
        self._state = state
    @property
    def name(self) -> str:
        return self._name
        # split = osp.split(self.path)
        # if not len(split):
        #     return '<Corrupted path>'
        # return split[-1]
    




class Dataset:
    def __init__(self, working_dir = None) -> None:
        self._working_dir : Optional[str] = working_dir
        self._doc_refs : list[DocumentReference] = []

        self.use_intermid_dirs: bool = True


    @property
    def working_dir(self) -> Optional[str]:
        return self._working_dir
    
    @working_dir.setter
    def working_dir(self, path: Union[Path,str]):
        path = str(path)
        self._working_dir = path

    def update_references(self):
        return
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
            self._doc_refs.append(DocumentReference(osp.split(fp)[-1],fp))
    @property
    def doc_refs(self):
        for ref in self._doc_refs:
            yield ref

    def create_new_reference(self, doc: Document) -> None:
        """
        This creates a refernce to a new in-memory doc, that is not yet saved on disk
        """

        doc_ref = DocumentReference(doc.name, None, DocumentState.UNSAVED)
        self._doc_refs.append(doc_ref)
        
        

        self.update_references()
    
