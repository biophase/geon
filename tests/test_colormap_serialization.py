import h5py
import numpy as np
import pytest

from geon.data.definitions import ColorMap
from geon.data.pointcloud import FieldBase, FieldType


@pytest.mark.parametrize('legacy', [False, True])
def test_color_map_round_trip_and_update(tmp_path, legacy):
    path = tmp_path / 'cloud.h5'
    values = np.arange(10, dtype=np.float32)
    original = ColorMap('original')
    with h5py.File(path, 'w') as handle:
        group = FieldBase('scalar', values, FieldType.SCALAR, original).save_hdf5(handle)
        if legacy:
            group.move('color_map', 'ColorMap')

    with h5py.File(path, 'r+') as handle:
        group = handle['scalar']
        loaded = FieldBase.from_hdf5_fieldgroup(group)
        assert loaded.color_map.name == 'original'
        np.testing.assert_array_equal(loaded.color_map.color_definitions, original.color_definitions)
        point_data_address = h5py.h5o.get_info(group['data'].id).addr

        updated = ColorMap('updated', 'hsv')
        updated.color_definitions[:] = [[0.2, 0.5, 1], [0.8, 1, 1]]
        updated.save_h5py(group)
        assert 'ColorMap' not in group
        map_address = h5py.h5o.get_info(group['color_map'].id).addr
        updated.save_h5py(group)
        assert h5py.h5o.get_info(group['color_map'].id).addr == map_address
        assert group['color_map'].attrs['columns'] == 'pos h s v'

        # A different number of stops (and dtype) replaces only the map.
        resized = ColorMap.get_cmap('viridis', (-5, 20))
        resized.save_h5py(group)
        assert h5py.h5o.get_info(group['data'].id).addr == point_data_address
        np.testing.assert_array_equal(group['data'][()], values)

    with h5py.File(path, 'r') as handle:
        loaded = FieldBase.from_hdf5_fieldgroup(handle['scalar'])
        assert loaded.color_map.name == resized.name
        assert loaded.color_map.color_type == resized.color_type
        np.testing.assert_array_equal(loaded.color_map.color_positions, resized.color_positions)
        np.testing.assert_array_equal(loaded.color_map.color_definitions, resized.color_definitions)
