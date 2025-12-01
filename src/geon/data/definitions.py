from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import h5py


@dataclass
class ColorMap:
    name: str
    color_type: Literal['rgb', 'hsv'] = 'rgb'
    color_positions:    NDArray[np.float32] = np.array([0.,1]) 
    color_definitions:  NDArray[np.float32] = np.array([[1.,0.,0.],[0.,1.,0.]])

    
    def save_h5py(self, field_group: h5py.Group) -> None:
        """
        write a dataset describing the cmap to a h5py group
        """
        data = np.concat([
            self.color_positions[:,None],
            self.color_definitions
        ], axis=1)
        dataset = field_group.create_dataset('ColorMap', data=data)
        dataset.attrs['name'] = self.name
        dataset.attrs['color_type'] = self.color_type
        dataset.attrs['columns'] = ' '.join(['pos'] + list(self.color_type))


    @classmethod
    def load_h5py(cls, dataset: h5py.Dataset):
        name = dataset.attrs.get('name',"Unnamed Colormap")
        color_type = dataset.attrs.get('color_type', 'rgb')
        assert color_type in ['hsv','rgb']
        data = dataset[()]
        assert isinstance(data , np.ndarray), f"Wrong parsing of colormap data, got: {type(data)}"
        return cls(
            name=name, 
            color_type=color_type, 
            color_positions=data[:,0], 
            color_definitions=data[:,1:]
            )
    
        


        
