import numpy as np
from .registry import register_data
from .base import BaseData


from typing import Tuple, TypedDict, List, Dict, Optional, Literal
from dataclasses import dataclass
import json



@register_data
class PointCloudData(BaseData):
    def __init__(self, points: np.ndarray):
        super().__init__()
        self.points = points
        self.segmentations : list[IndexSegmentationBase] = []


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

    def reindex(self) -> None: # FIXME: should return a mapping from old to new
        self.semantic_classes.sort(key = lambda x: x.id)
        for i, s in enumerate(self.semantic_classes):
            if s.id != -1:
                s.id = i    


class IndexSegmentationBase:
    raise NotImplementedError
    pass