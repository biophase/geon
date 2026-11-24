import weakref
from unittest.mock import Mock

import numpy as np
import pytest

from geon.data.boundingbox import BoundingBox, BoundingBoxData
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.tools.boundingbox import AddBoundingBoxesCmd
from geon.tools.boundingbox import CreateHorizontalBoundingBoxTool


def test_first_edge_preview_does_not_rescan_scene_bounds():
    tool = object.__new__(CreateHorizontalBoundingBoxTool)
    tool._active_scene_bounds = Mock(return_value=(0, 100, 0, 40, -2, 8))
    tool.c1 = np.array([0., 0., 0.])
    tool._init_vertical_extents()
    for x in (1., 2., 10.):
        box = tool._make_stage2_preview_box(np.array([x, 0., 0.]))
        assert box.width == pytest.approx(x)
        assert box.depth == pytest.approx(0.25)
        assert box.height == pytest.approx(10.)
    tool._active_scene_bounds.assert_called_once()
    # A subsequent activation refreshes the cached scale, including empty scenes.
    tool._active_scene_bounds.return_value = None
    tool._init_vertical_extents()
    box = tool._make_stage2_preview_box(np.array([1., 0., 0.]))
    assert box.depth == pytest.approx(0.0025)


def test_add_bounding_boxes_command_is_atomic_for_undo():
    layer = BoundingBoxLayer(BoundingBoxData([]))
    boxes = [BoundingBox(id="a"), BoundingBox(id="b")]
    command = AddBoundingBoxesCmd("Add boxes", weakref.ref(layer), boxes)

    command.execute()
    assert [box.id for box in layer.data.boxes] == ["a", "b"]

    command.undo()
    assert layer.data.boxes == []

    command.execute()
    assert [box.id for box in layer.data.boxes] == ["a", "b"]
