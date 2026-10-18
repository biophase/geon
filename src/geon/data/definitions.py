from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

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
    min: Optional[float] = None
    max: Optional[float] = None

    def validate(self):
        validate_stops(self.color_positions, self.color_definitions, self.min, self.max)
        if np.asarray(self.color_definitions).shape != (len(self.color_positions), 3):
            raise ValueError('Color stops require three components')
        if self.color_type not in ('rgb', 'hsv'):
            raise ValueError('Unknown color space')

    def signature(self):
        self.validate()
        return ('scalar', self.color_type, self.min, self.max,
                tuple(self.color_positions), tuple(map(tuple, self.color_definitions)))

    
    
    def save_h5py(self, field_group: h5py.Group) -> None:
        """Save or update only the color map in an existing field group."""
        self.validate()
        data = np.concat([
            self.color_positions[:,None],
            self.color_definitions
        ], axis=1)
        dataset = field_group.get('color_map')
        if isinstance(dataset, h5py.Dataset) and dataset.shape == data.shape and dataset.dtype == data.dtype:
            dataset[...] = data
        else:
            if dataset is not None:
                del field_group['color_map']
            dataset = field_group.create_dataset('color_map', data=data)
        dataset.attrs['name'] = self.name
        dataset.attrs['color_type'] = self.color_type
        dataset.attrs['columns'] = ' '.join(['pos'] + list(self.color_type))
        dataset.attrs['mapping_type'] = 'scalar'
        save_limits(dataset, self)
        # Older versions saved this spelling, but did not load it correctly.
        if 'ColorMap' in field_group:
            del field_group['ColorMap']


    @classmethod
    def load_h5py(cls, dataset: h5py.Dataset):
        if isinstance(dataset, h5py.Group):
            return VectorColorMap.load_h5py(dataset)
        name = dataset.attrs.get('name',"Unnamed Colormap")
        color_type = dataset.attrs.get('color_type', 'rgb')
        name = name.decode('utf-8') if isinstance(name, bytes) else str(name)
        color_type = color_type.decode('utf-8') if isinstance(color_type, bytes) else str(color_type)
        assert color_type in ['hsv','rgb']
        data = dataset[()]
        assert isinstance(data , np.ndarray), f"Wrong parsing of colormap data, got: {type(data)}"
        positions = data[:, 0]
        colors = data[:, 1:]
        minimum, maximum = read_limits(dataset)
        if 'mapping_type' not in dataset.attrs:
            order = np.argsort(positions, kind='stable')
            positions, colors = positions[order], colors[order]
            positions, indices = np.unique(positions, return_index=True)
            colors = colors[indices]
            minimum, maximum = float(positions[0]), float(positions[-1])
            if minimum == maximum:
                positions, colors = np.array([0., 1.]), np.repeat(colors[-1:], 2, axis=0)
            else:
                positions = (positions - minimum) / (maximum - minimum)
        result = cls(
            name=name, 
            color_type=color_type, 
            color_positions=positions,
            color_definitions=colors, min=minimum, max=maximum,
            )
        result.validate()
        return result
    
    @classmethod
    def get_cmap(cls, cmap_type: Optional[str], min_max: Optional[Tuple[float, float]] = None) -> "ColorMap":

        
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

        if min_max is not None:
            cmap.min, cmap.max = map(float, min_max)

        return cmap


def validate_stops(positions, outputs, minimum, maximum):
    p, c = np.asarray(positions), np.asarray(outputs)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('At least two finite stops are required')
    if p[0] != 0 or p[-1] != 1 or np.any(np.diff(p) <= 0):
        raise ValueError('Stops must increase strictly from 0 to 1')
    if len(c) != len(p) or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError('Stop outputs must be finite and between 0 and 1')
    for limit in (minimum, maximum):
        if limit is not None and not np.isfinite(limit):
            raise ValueError('Limits must be finite or empty')
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError('Minimum must not exceed maximum')


def save_limits(obj, mapping):
    for key in ('min', 'max'):
        value = getattr(mapping, key)
        if key in obj.attrs:
            del obj.attrs[key]
        if value is not None:
            obj.attrs[key] = float(value)


def read_limits(obj):
    return tuple(float(obj.attrs[k]) if k in obj.attrs else None for k in ('min', 'max'))


@dataclass
class ChannelMap:
    color_positions: np.ndarray = field(default_factory=lambda: np.array([0., 1.]))
    output_values: np.ndarray = field(default_factory=lambda: np.array([0., 1.]))
    min: Optional[float] = None
    max: Optional[float] = None

    def validate(self):
        validate_stops(self.color_positions, self.output_values, self.min, self.max)
        if np.asarray(self.output_values).ndim != 1:
            raise ValueError('Channel outputs must be one-dimensional')

    def signature(self):
        self.validate()
        return (self.min, self.max, tuple(self.color_positions), tuple(self.output_values))


@dataclass
class VectorColorMap:
    name: str
    red: ChannelMap = field(default_factory=ChannelMap)
    green: ChannelMap = field(default_factory=ChannelMap)
    blue: ChannelMap = field(default_factory=ChannelMap)

    @property
    def channels(self):
        return (self.red, self.green, self.blue)

    def validate(self):
        for channel in self.channels:
            channel.validate()

    def signature(self):
        return ('vector', *(channel.signature() for channel in self.channels))

    def save_h5py(self, field_group):
        self.validate()
        for key in ('color_map', 'ColorMap'):
            if key in field_group:
                del field_group[key]
        group = field_group.create_group('color_map')
        group.attrs['mapping_type'] = 'vector'
        group.attrs['name'] = self.name
        for key, channel in zip(('red', 'green', 'blue'), self.channels):
            ds = group.create_dataset(key, data=np.column_stack((channel.color_positions, channel.output_values)))
            save_limits(ds, channel)

    @classmethod
    def load_h5py(cls, group):
        if group.attrs.get('mapping_type') != 'vector':
            raise ValueError('Unknown color mapping type')
        channels = []
        for key in ('red', 'green', 'blue'):
            ds = group[key]
            values = ds[()]
            channels.append(ChannelMap(values[:, 0], values[:, 1], *read_limits(ds)))
        result = cls(group.attrs['name'], *channels)
        result.validate()
        return result


def vector_preset(minimum, maximum):
    return VectorColorMap(f'RGB ({minimum} to {maximum})', *[
        ChannelMap(min=minimum, max=maximum) for _ in range(3)])


def finite_extrema(values):
    values = np.asarray(values)
    low, high = np.inf, -np.inf
    # Bound temporary storage for large clouds, including strided components.
    flat = values.flat
    for start in range(0, values.size, 1_000_000):
        block = np.asarray(flat[start:start+1_000_000])
        if np.issubdtype(block.dtype, np.integer):
            low, high = min(low, block.min()), max(high, block.max())
            continue
        valid = np.isfinite(block)
        low = min(low, np.min(block, where=valid, initial=np.inf))
        high = max(high, np.max(block, where=valid, initial=-np.inf))
    return (float(low), float(high)) if low != np.inf else (0., 1.)


def resolve_limits(mapping, extrema):
    low = extrema[0] if mapping.min is None else mapping.min
    high = extrema[1] if mapping.max is None else mapping.max
    if low > high:
        raise ValueError('Resolved minimum exceeds maximum')
    return float(low), float(high)


def sample_map(mapping, positions):
    """Evaluate relative positions identically for previews and GPU tables."""
    mapping.validate()
    t = np.clip(positions, 0, 1)
    if isinstance(mapping, ChannelMap):
        return np.interp(t, mapping.color_positions, mapping.output_values)
    # VTK supplies the same HSV interpolation convention as existing rendering.
    import vtk
    ctf = vtk.vtkColorTransferFunction()
    if mapping.color_type == 'hsv':
        ctf.SetColorSpaceToHSV()
    for position, color in zip(mapping.color_positions, mapping.color_definitions):
        if mapping.color_type == 'hsv':
            ctf.AddHSVPoint(float(position), *map(float, color))
        else:
            ctf.AddRGBPoint(float(position), *map(float, color))
    if mapping.color_type == 'rgb':
        return np.stack([np.interp(t, mapping.color_positions, mapping.color_definitions[:, c])
                         for c in range(3)], axis=-1)
    from vtk.util import numpy_support as ns
    values = ns.numpy_to_vtk(np.ascontiguousarray(t).reshape(-1), deep=False)
    rgb = ns.vtk_to_numpy(ctf.MapScalars(values, vtk.VTK_COLOR_MODE_MAP_SCALARS, 0))[:, :3]
    return (rgb.astype(float) / 255.).reshape(np.shape(t) + (3,))


        
PARAM_MAP: Dict[str, ColorMap] = {

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
                    [0.3, 0.3, 0.3],
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
