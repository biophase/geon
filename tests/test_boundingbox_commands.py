import weakref

from geon.data.boundingbox import BoundingBox, BoundingBoxData
from geon.rendering.boundingbox import BoundingBoxLayer
from geon.tools.boundingbox import AddBoundingBoxesCmd


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
