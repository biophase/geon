import os
import os.path as osp
from glob import glob
from pathlib import Path
import h5py

from geon.data.document import Document
from geon.data.pointcloud import SemanticSchema, PointCloudData, SemanticSegmentation

from geon.version import GEON_FORMAT_VERSION
from typing import Union, Optional, cast, Callable
from dataclasses import dataclass
from enum import Enum, auto
# import traceback
from copy import deepcopy



class RefModState(Enum):
    UNSPECIFIED = auto()
    MODIFIED    = auto()
    SAVED       = auto()

class RefLoadedState(Enum):
    UNSPECIFIED =auto()
    REFERENCE   =auto()
    LOADED      =auto()
    ACTIVE      =auto()

@dataclass
class DocumentReference:
    _name: str
    _path: Optional[str]
    _modState: RefModState      = RefModState.UNSPECIFIED
    _loadedState:RefLoadedState = RefLoadedState.UNSPECIFIED

    # @classmethod
    # def create(cls, path: str) -> "DocumentReference":
    #     return cls(path, DocumentState.MODIFIED)
    
    def load(self) -> Document:
        if self.path is None:
            raise Exception("Attempted to load a reference with no path")
        doc = Document.load_hdf5(self.path)
        self._modState = RefModState.SAVED
        return doc
    
    @property
    def path(self) -> Optional[str]:
        return self._path
    
    @path.setter
    def path(self, path:str)->None:
        self._path = path
        
    @property
    def modState(self) -> RefModState:
        return self._modState

    @modState.setter
    def modState(self, state: RefModState) -> None:
        self._modState = state
        
    @property
    def loadedState(self) -> RefLoadedState:
        return self._loadedState
    
    @loadedState.setter
    def loadedState(self, state: RefLoadedState) -> None:
        self._loadedState = state

    @property
    def name(self) -> str:
        return self._name
        # split = osp.split(self.path)
        # if not len(split):
        #     return '<Corrupted path>'
        # return split[-1]


class Dataset:
    """
    The task of the dataset is two-fold:
        1. It manages references to documents on disk
        2. It holds onto loaded documents for modifications by a user
    """
    def __init__(self, working_dir = None) -> None:
        self._working_dir:      Optional[str] = working_dir
        self._doc_refs:         list[DocumentReference] = []
        self._loaded_docs:      dict[str, Document] = {}
        self._max_loaded_docs:  int = 5


        self.use_intermid_dirs: bool = True


    @property
    def working_dir(self) -> Optional[str]:
        return self._working_dir
    
    def add_document(self, doc: Document) -> DocumentReference:
        """
        add a constructed document and create the reference for it
        """
        assert doc.name not in self.doc_ref_names
        doc_ref = self.create_new_reference(doc)
        self._loaded_docs[doc.name] = doc
        self.pop_old_loaded()
        return doc_ref
        
        
        
        
    def pop_old_loaded(self) -> None:
        if len(self._loaded_docs) > self._max_loaded_docs:
                oldest_doc_name = next(iter(self._loaded_docs))
                self._loaded_docs.pop(oldest_doc_name)
                for r in self.doc_refs:
                    if r.name == oldest_doc_name:
                        r._loadedState = RefLoadedState.REFERENCE
                        
    
    def activate_reference(self, doc_ref: DocumentReference) -> Document:
        """ load doc and move active pointer to reference """
        doc = self._load_reference(doc_ref)
        doc_ref.loadedState = RefLoadedState.ACTIVE
        return doc
    
    def deactivate_current_ref(self) -> None:
        for ref in self.doc_refs:
            if ref.loadedState == RefLoadedState.ACTIVE:
                ref._loadedState = RefLoadedState.LOADED
    
    def _load_reference(self, doc_ref: DocumentReference) -> Document:
        
        pass
        # check if already loaded and move to front
        if doc_ref.name in self._loaded_docs.keys():
            doc = self._loaded_docs.pop(doc_ref.name)
            self._loaded_docs[doc.name] = doc
            return doc
        else:
            doc = doc_ref.load()
            doc_ref._loadedState = RefLoadedState.LOADED
            self._loaded_docs[doc.name] = doc
            self.pop_old_loaded()
            return doc
            
    
    def _unload_reference(self, doc_ref: DocumentReference):
        # check if at all loaded
        if doc_ref.name not in self._loaded_docs.keys():
            return
        else:
            doc = self._loaded_docs.pop(doc_ref.name)
            doc_ref._loadedState = RefLoadedState.REFERENCE
            print(f"Unloaded document {doc.name}")
            
    
    @working_dir.setter
    def working_dir(self, path: Union[Path,str]):
        path = str(path)
        self._working_dir = path

    def populate_references(self):
        # self._temp +=1
        # if self._temp > 1:
        #     raise Exception
        
        # return
        if self.working_dir is None:
            return
        
        # self._doc_refs.clear()
        file_paths = list(glob(osp.join(self.working_dir, "*.hdf5")))
        file_paths+= list(glob(osp.join(self.working_dir, "*.h5")))
        file_paths+= list(glob(osp.join(self.working_dir, "*", "*.hdf5")))
        file_paths+= list(glob(osp.join(self.working_dir, "*", "*.h5")))
        
        print(f'called populate references {file_paths=}')
        # traceback.print_stack()
        for fp in file_paths:

            with h5py.File(fp, "r") as f:
                version = f["document"].attrs["geon_format_version"]
                assert not isinstance(version,  h5py.Empty)
                if version.astype(int) > GEON_FORMAT_VERSION:
                    raise ValueError(
                        f"Unsupported GEON format version {version} in {fp}; "
                        f"current version is {GEON_FORMAT_VERSION}"
                    )
                raw_name = f['document'].attrs.get('name')
                doc_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            doc_ref = DocumentReference(
                # osp.split(fp)[-1],
                doc_name,
                fp,
                RefModState.SAVED,
                RefLoadedState.REFERENCE
                )
            if doc_ref.name not in self.doc_ref_names:
                self._doc_refs.append(doc_ref)
                
    
    
    def _get_semantic_schemas(self) -> tuple[dict[str,SemanticSchema], dict[str,SemanticSchema]]:

        referenced_schemas  : dict[str, SemanticSchema] = dict()
        loaded_schemas      : dict[str, SemanticSchema] = dict()

        for ref in self.doc_refs:
            doc = self._loaded_docs.get(ref.name)
            if doc is not None:
            # # gather schemas from loaded docs
                   
                
                for data_name, data in doc.scene_items.items():
                    if isinstance(data, PointCloudData):
                        for field in data.get_fields():
                            if isinstance (field, SemanticSegmentation):
                                build_key = f"{ref.name}/{data_name}/{field.name}/{field.schema.name}"
                                loaded_schemas[build_key] = field.schema
                
            # gather schemas from referenced documents
            else:
                assert ref.path is not None
                referenced_schemas = referenced_schemas | SemanticSchema.scan_h5(ref.path)
        pass                
        return referenced_schemas, loaded_schemas
    
    def get_matching_schemas(
        self,
        schema: SemanticSchema
    ) -> dict[str, SemanticSchema]:
        
        
        schemas_matching: dict[str, SemanticSchema]= dict()
        r_schemas, l_schemas = self._get_semantic_schemas()
        schemas = r_schemas | l_schemas
        
        schemas_matching: dict[str, SemanticSchema]= dict()
        for bk, schema in schemas.items():
            if schema.signature() == schema.signature() and\
                schema.name == schema.name:
                    schemas_matching[bk] = schema
        
        return schemas_matching
    
    def update_semantic_schema (
        self,
        old_schema: SemanticSchema,
        new_schema: SemanticSchema,
        old_2_new_ids: list[tuple[int,int]],
        progress_cb: Optional[Callable[[int, int, str],None]] = None
        ) -> list[DocumentReference]:
        """
        Replace matching semantic schemas across referenced/loaded documents, 
        remap labels, and save changes.
        """
        
        schemas_matching = self.get_matching_schemas(old_schema)
        
        active_ref = list(
            [ref for ref in self.doc_refs if 
             ref.loadedState == RefLoadedState.ACTIVE])
    
        saved_refs = []
        # references:
        tot_len = len(schemas_matching)
        for i, (build_key, schema) in enumerate(schemas_matching.items()):
            

            if progress_cb is not None:
                progress_cb(i, tot_len, build_key)
                
            r_name, data_name, field_name, schema_name = build_key.split('/')
            ref = list([r for r in self.doc_refs if r.name == r_name])[0]
            
            doc = self._loaded_docs.get(ref.name)
            if doc is None:
                doc = self._load_reference(ref)
            
            data = cast(PointCloudData, doc.scene_items.get(data_name))
            field = cast(SemanticSegmentation, data.get_fields(field_name)[0])
            field.remap(old_2_new_ids)
            field.schema = new_schema
            if ref.path is None:
                continue
            doc.save_hdf5(ref.path)
            saved_refs.append(ref)
        return saved_refs
                    

    @property
    def unique_semantic_schemas(self) -> list[SemanticSchema]:
        out = {}
        # get all schemas in the document
        schemas = self._get_semantic_schemas()
        schemas = schemas[0] | schemas[1] # referenced and loaded
        
        # reduce to unique
        for schema in schemas.values():
            if schema.signature() not in out.keys():
                out[schema.signature()] = deepcopy(schema)
                
        out = list(out.values())
        return out
        
            
    @property
    def doc_refs(self):
        for ref in self._doc_refs:
            yield ref
            
    @property
    def doc_ref_names(self) -> list[str]:
        return [ref.name for ref in self.doc_refs]
    def create_new_reference(self, doc: Document) -> DocumentReference:
        """
        This creates a refernce to a new in-memory doc, that is not yet saved on disk
        """

        doc_ref = DocumentReference(doc.name, None, RefModState.MODIFIED)
        for ref in self.doc_refs:
            if doc_ref.name == ref:
                raise ValueError(f"Attempted to add a DocumentReference with duplicate names: {doc_ref.name}")
        self._doc_refs.append(doc_ref)
        return doc_ref
        

    
