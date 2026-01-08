from enum import Enum, auto
from geon.util.resources import resource_path
class Boolean(Enum):
    UNION = auto()
    DIFFERENCE = auto()
    INTERSECTION = auto()
    EXCLUSION = auto()
    OVERWRITE = auto()
    
    @staticmethod
    def icon_path(bool_type: "Boolean") -> str:
        return {
            Boolean.UNION : resource_path('bool_union.png'),
            Boolean.DIFFERENCE : resource_path('bool_difference.png'),
            Boolean.INTERSECTION : resource_path('bool_intersection.png'),
            Boolean.EXCLUSION : resource_path('bool_exclusion.png'),
            Boolean.OVERWRITE : resource_path('bool_overwrite.png')
            
        }[bool_type]
        
        
    @property
    def default(self) -> "Boolean":
        return Boolean.OVERWRITE
