import numpy as np
import json
from typing import Tuple, TypedDict, List, Dict, Optional, Literal
import pickle


class SemanticDescription(TypedDict):
    name: str
    color:Tuple[float,float,float]


    

class SemanticClass:
    def __init__(self, id:int, name:str, color:Tuple[float,float,float]):
        self.id = id
        self.name = name
        self.color = color

    @property
    def as_id_dict(self)->Dict[int, SemanticDescription]:
        return {self.id : SemanticDescription(name=self.name, color=self.color)}
    
    def __hash__(self) -> int:
        return hash((self.id, self.name, self.color))


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
    def from_dict(cls, semantic_dict : Dict[int, SemanticDescription]) -> "SemanticSchema":
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
        assert semantic_class.id not in [s.id for s in self.semantic_classes], f"Index {semantic_class.id} already exists in schema."
        assert semantic_class.id not in [s.name for s in self.semantic_classes], f"Name {semantic_class.name} already exists in schema."
        self.semantic_classes.append(semantic_class)
        self.semantic_classes.sort(key = lambda x: x.id)

    def remove_semantic_class(self, id:int) -> None:
        assert id in [s.id for s in self.semantic_classes], f"Index {id} is does not exist and can't be deleted."
        self.semantic_classes = [s for s in self.semantic_classes if s.id != id]

    def reindex(self) -> None: # FIXME: should return a mapping from old to new
        self.semantic_classes.sort(key = lambda x: x.id)
        for i, s in enumerate(self.semantic_classes):
            if s.id != -1:
                s.id = i    

class InstanceStatistic(TypedDict):
    number_of_instances : int
    amount_of_instance_points : Tuple[int, float]
    amount_of_stuff_points : Tuple [int,float]

class SegmentationStatistic(TypedDict):
    semantic_statistic : Optional[Dict[int, Tuple[int, float]]] # maps class index to `total number` and `percentage`
    instance_statistic : Optional[InstanceStatistic]

class IndexSegmentation:
    """ 
    A point cloud segmentation that holds instance and semantic information for every
    point. All indices are strictly positive or 0. Lack of information is labeled as -1.
    
    
    """
    def __init__(self, name:str, size:int, store_semantic_idx = True, store_instance_idx = True):
        self.name = name
        self.semantic_idx = np.ones(size, dtype = int) * -1 if store_semantic_idx else None
        self.instance_idx = np.ones(size, dtype = int) * -1 if store_instance_idx else None
        self.semantic_schema = SemanticSchema()

    def get_statistic(self)->SegmentationStatistic:
        if self.semantic_idx is not None:
            sem_inds, sem_counts = np.unique(self.semantic_idx, return_counts=True)
            sem_statistic = {idx : (count, count / len(self.semantic_idx)) for idx, count in zip(sem_inds, sem_counts)}
        else:
            sem_statistic = None

        if self.instance_idx is not None:
            instance_inds = np.unique(self.instance_idx)
            instance_points = np.sum(self.instance_idx != -1)
            instance_statistic = InstanceStatistic(
                number_of_instances = len(instance_inds),
                amount_of_instance_points = (instance_points, instance_points / len(self.instance_idx)),
                amount_of_stuff_points = (len(self.instance_idx) - instance_points, (1 - instance_points / len(self.instance_idx)))
            )
        else:
            instance_statistic = None

        return SegmentationStatistic(
            semantic_statistic = sem_statistic,
            instance_statistic = instance_statistic
        )
    def get_next_instance_id(self)->int:

        if self.instance_idx is not None:
            nonneg = self.instance_idx[self.instance_idx >= 0]
            size = len(self.instance_idx) + 1
            present = np.zeros(size, dtype=bool)
            present[nonneg[nonneg < size]] = True
            return np.flatnonzero(~present)[0]
        else:
            raise Exception("The current schema does not support instance indexing")
        
    def remove_semantic_class (self, id:int) -> None:
        self.semantic_schema.remove_semantic_class(id)
        if self.semantic_idx is not None:
            self.semantic_idx[self.semantic_idx == id] = -1

    def replace_semantic_schema (self, new_schema : SemanticSchema, by : Literal['id','name'] = 'id') -> None:
        if by=='id':
            for s_old in self.semantic_schema.semantic_classes:
                if s_old.id not in [s.id for s in new_schema.semantic_classes]:
                    self.remove_semantic_class(s_old.id)
        elif by == 'name':
            raise NotImplementedError # TODO

        self.semantic_schema = new_schema

    def reindex_semantic(self):
        raise NotImplementedError # TODO: should reindex the containers, not only the schema. requires mapping output from schema method
    
    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path) -> "IndexSegmentation":
        with open(file_path, "rb") as f:
            seg = pickle.load(f)
        return seg


        

    