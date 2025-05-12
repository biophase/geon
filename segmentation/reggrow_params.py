from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray


from typing import Annotated
from numpy .typing import NDArray
import numpy as np

FloatNx3 = Annotated[NDArray[np.float32], ("N", 3)]
FloatNxM = Annotated[NDArray[np.float32], ("N","M")]
FloatN = Annotated[NDArray[np.float32], ("N")]
IntN = Annotated[NDArray[np.int32], ("N")]

class BuildParams:
    def __init__(self, **kwargs):
            self.epsilon=0.03
            self.refit_multiplier=2.0
            self.epsilon_multiplier=3.0  # Value in Poux & Kobbelt paper = 3
            self.epsilon_multiplier_average=1.5  # average residual should be below value * epsilon, otherwise stop growing
            self.search_radius_approx=0.0  # accuracy of radius search; 0. = exact search
            self.min_points_in_region=80
            self.first_refit=4
            self.alpha = np.pi * 0.15  # PI * 0.15
            self.min_success_ratio=0.10
            self.max_dist_from_cent=50.0
            self.tracker_size=50
            self.oriented_normals=False
            self.verbose=True
            self.perform_cca=True  # divides each region in connected components after segmentation
            self.cleanup_small_edges=True
            self.perform_cca_on_remaining=False

            for k, v in kwargs.items():
                if not hasattr(self, k):
                    raise TypeError(f'{self.__class__.__name__} got an unexpected parameters {k!r}')
                setattr(self, k, v)
            
            
    def __repr__(self):
        return str(self.__dict__)
    
    def to_dict(self)-> Dict:
        return self.__dict__
    
class ReggrowParams:
    def __init__(self,
                 chunk_size_x:int = 1,
                 chunk_size_y:int = 1,
                 chunk_size_z:int = 1,
                 chunk_targetSize_x:int = 5,
                 chunk_targetSize_y:int = 5,              
                 chunk_targetSize_z: Optional[int] = None,
                 **kwargs
                 ):
        self.chunk_size:NDArray[np.int32] = np.array([chunk_size_x,
                                                      chunk_size_y,
                                                      chunk_size_z
                                                      ])
        self.build_params = BuildParams(**kwargs)
        self.chunk_targetSize_x = chunk_targetSize_x
        self.chunk_targetSize_y = chunk_targetSize_y
        self.chunk_targetSize_z = chunk_targetSize_z
    
    def __repr__(self):
        return str(dict(
            chunks_size = self.chunk_size,
            build_params = self.build_params
        ))
        
    @classmethod
    def from_dict(cls, data_dict) -> "ReggrowParams":
        data = ReggrowParams()
        data.chunk_size = data_dict['chunk_size']
        data.chunk_targetSize_x = data_dict['chunk_targetSize_x']
        data.chunk_targetSize_y = data_dict['chunk_targetSize_y']
        data.chunk_targetSize_z = data_dict['chunk_targetSize_z']
        data.build_params = BuildParams()
        for k,v in data_dict['build_params']:
            data.build_params.__setattr__(k,v)
        return data
    
    def set_uniform_chunks (self, pcd_xyz:FloatNx3) -> None:
        if self.chunk_targetSize_z is not None:
            self._set_uniform_chunks_xyz(pcd_xyz)
        else: self._set_uniform_chunks_xy (pcd_xyz)
    
    def _set_uniform_chunks_xy (self, pcd_xyz:FloatNx3):
        self.chunk_size[0] = int((pcd_xyz[:,0].max() - pcd_xyz[:,0].min()) / self.chunk_targetSize_x)
        self.chunk_size[1] = int((pcd_xyz[:,1].max() - pcd_xyz[:,1].min()) / self.chunk_targetSize_y)
        # if self.build_params.verbose:
        print(f'Set chunk size X to {self.chunk_size[0]} and Y to {self.chunk_size[1]}')
    
    def _set_uniform_chunks_xyz(self, pcd_xyz:FloatNx3):
        assert self.chunk_targetSize_z is not None
        raise NotImplementedError