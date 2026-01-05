from enum import Enum, auto
class Boolean(Enum):
    UNION = auto()
    DIFFERENCE = auto()
    INTERSECTION = auto()
    EXCLUSION = auto()
    OVERWRITE = auto()
    
    @staticmethod
    def icon_path(bool_type: "Boolean") -> str:
        return {
            Boolean.UNION : 'resources/bool_union.png',
            Boolean.DIFFERENCE : 'resources/bool_difference.png',
            Boolean.INTERSECTION : 'resources/bool_intersection.png',
            Boolean.EXCLUSION : 'resources/bool_exclusion.png',
            Boolean.OVERWRITE : 'resources/bool_overwrite.png'
            
        }[bool_type]
        
        
    @property
    def default(self) -> "Boolean":
        return Boolean.OVERWRITE