from dataclasses import dataclass, field

import numpy as np

import h5py

from typing import List, Dict

@dataclass
class Plane:
    origin: np.ndarray = field(default_factory=lambda: np.array([0,0,0]))
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1,0,0]))
    y_axis: np.ndarray = field(default_factory=lambda: np.array([0,1,0]))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0,0,1]))
    
    def __post_init__(self):
        # TODO: normalization, orthogonality checks, etc.
        ...
        
    @classmethod
    def abcd (a: float, b:float, c:float, d: float)->"Plane":
        """
        constructs a plane from an equation
        ax + by + cz +d = 0
        """
        ...
        # TODO
        
    @classmethod
    def from_points (origin: np.ndarray, x_point :np.ndarray, y_point: np.ndarray)->"Plane":
        ...
        # TODO
        
    def load_hdf5(cls, group: h5py.Group):
        ...
        # TODO
    def save_hdf5(self, group: h5py.Group) -> h5py.Group:
        ...
        # TODO
        
        