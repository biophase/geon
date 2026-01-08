from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
import h5py

npa = np.array



@dataclass
class ColorMap:
    name: str
    color_type: Literal['rgb', 'hsv'] = 'rgb'
    color_positions:    NDArray[np.float32] = field(default_factory=lambda: np.array([0., 1.], dtype=np.float32))
    color_definitions:  NDArray[np.float32] = field(default_factory=lambda: np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32))

    
    
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
    
    @classmethod
    def get_cmap(cls, cmap_type: Optional[str], min_max: tuple[float, float]) -> "ColorMap":

        
        cmap_type = cmap_type or 'rainbow'

        if cmap_type not in PARAM_MAP:
            raise KeyError(f"Unknown colormap '{cmap_type}'")

        #copy so we don't mutate the template
        base = PARAM_MAP[cmap_type]
        cmap = cls(
            name=base.name,
            color_type=base.color_type,
            color_positions=base.color_positions.copy(),
            color_definitions=base.color_definitions.copy(),
        )

        dif = min_max[1] - min_max[0]
        cmap.color_positions = cmap.color_positions * dif + min_max[0]

        return cmap


        
PARAM_MAP: dict[str, ColorMap] = {

            # =========================
            # Basic maps
            # =========================
            'red-green': ColorMap(
                name='red-green',
                color_type='rgb',
                color_positions=npa([0.0, 1.0]),
                color_definitions=npa([
                    [1., 0., 0.],
                    [0., 1., 0.],
                ])
            ),

            'gray': ColorMap(
                name='gray',
                color_type='rgb',
                color_positions=npa([0.0, 1.0]),
                color_definitions=npa([
                    [0., 0., 0.],
                    [1., 1., 1.],
                ])
            ),

            # =========================
            # Scientific / perceptual
            # =========================
            'viridis': ColorMap(
                name='viridis',
                color_type='rgb',
                color_positions=npa([0.0, 0.25, 0.5, 0.75, 1.0]),
                color_definitions=npa([
                    [0.267, 0.005, 0.329],
                    [0.283, 0.141, 0.458],
                    [0.254, 0.265, 0.530],
                    [0.207, 0.372, 0.553],
                    [0.993, 0.906, 0.144],
                ])
            ),

            'plasma': ColorMap(
                name='plasma',
                color_type='rgb',
                color_positions=npa([0.0, 0.25, 0.5, 0.75, 1.0]),
                color_definitions=npa([
                    [0.050, 0.030, 0.528],
                    [0.382, 0.089, 0.725],
                    [0.647, 0.165, 0.620],
                    [0.902, 0.387, 0.325],
                    [0.941, 0.975, 0.131],
                ])
            ),

            'inferno': ColorMap(
                name='inferno',
                color_type='rgb',
                color_positions=npa([0.0, 0.25, 0.5, 0.75, 1.0]),
                color_definitions=npa([
                    [0.002, 0.005, 0.014],
                    [0.258, 0.038, 0.406],
                    [0.578, 0.148, 0.404],
                    [0.902, 0.387, 0.161],
                    [0.988, 0.998, 0.645],
                ])
            ),

            'magma': ColorMap(
                name='magma',
                color_type='rgb',
                color_positions=npa([0.0, 0.25, 0.5, 0.75, 1.0]),
                color_definitions=npa([
                    [0.001, 0.000, 0.015],
                    [0.251, 0.073, 0.377],
                    [0.571, 0.186, 0.491],
                    [0.901, 0.444, 0.372],
                    [0.987, 0.991, 0.749],
                ])
            ),

            # =========================
            # Diverging
            # =========================
            'coolwarm': ColorMap(
                name='coolwarm',
                color_type='rgb',
                color_positions=npa([0.0, 0.5, 1.0]),
                color_definitions=npa([
                    [0.231, 0.298, 0.753],
                    [0.865, 0.865, 0.865],
                    [0.706, 0.016, 0.150],
                ])
            ),

            # =========================
            # Legacy / qualitative
            # =========================
            'rainbow': ColorMap(
                name='rainbow',
                color_type='rgb',
                color_positions=npa([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                color_definitions=npa([
                    [0.5, 0.0, 1.0],   # violet
                    [0.0, 0.0, 1.0],   # blue
                    [0.0, 1.0, 1.0],   # cyan
                    [0.0, 1.0, 0.0],   # green
                    [1.0, 1.0, 0.0],   # yellow
                    [1.0, 0.0, 0.0],   # red
                ])
            ),

            'jet': ColorMap(
                name='jet',
                color_type='rgb',
                color_positions=npa([0.0, 0.35, 0.5, 0.65, 1.0]),
                color_definitions=npa([
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.5, 0.0, 0.0],
                ])
            ),
        }
