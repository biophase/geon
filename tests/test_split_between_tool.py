from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.tools.base import Event
from geon.tools.command_manager import CommandManager
from geon.tools.split_between import SplitBetweenCommitCmd, SplitBetweenTool
from geon.tools.tool_context import ToolContext


class _Signal:
    def __init__(self):
        self.calls = 0

    def emit(self, *_args):
        self.calls += 1


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _fixture():
    data = PointCloudData(np.arange(24, dtype=np.float32).reshape(8, 3))
    data.add_field("instances", np.asarray([0, 0, 1, 1, 2, 2, 3, -1]), FieldType.INSTANCE)
    layer = PointCloudLayer(data)
    layer.active_field_name = "instances"
    signal = _Signal()
    controller = SimpleNamespace(
        layer_data_modified=signal,
        scene_tree_request_change=_Signal(),
        deactivate_tool=lambda: None,
    )
    viewer = SimpleNamespace(rerender=lambda: None, window=lambda: None, pick=lambda: None)
    scene = SimpleNamespace(active_layer=layer)
    ctx = ToolContext(scene=scene, viewer=viewer, controller=controller)
    tool = SplitBetweenTool(CommandManager(), ctx)
    return layer, controller, viewer, tool


def _event(*, shift=False):
    return Event((0, 0), (0, 0), shift, False, False, None)


def test_tool_defaults_to_active_instance_field_and_collects_visible_instances(qt_app):
    layer, _controller, viewer, tool = _fixture()
    layer._visibility_mask = np.asarray([True, False, True, True, True, True, True, True])
    tool.active_set = "work"
    viewer.pick = lambda: SimpleNamespace(layer=layer, element_idx=0)

    tool._pick_instance(_event(shift=True))

    assert tool.field_name == "instances"
    assert tool.point_sets["work"] == {0}
    assert tool.active_set == "work"
    assert layer._temporary_point_opacity[0] == pytest.approx(0.5)


def test_picking_toggles_and_moves_instances_between_sets(qt_app):
    layer, _controller, viewer, tool = _fixture()
    viewer.pick = lambda: SimpleNamespace(layer=layer, element_idx=2)
    tool.active_set = "a"
    tool._pick_instance(_event(shift=True))
    assert tool.point_sets["a"] == {2, 3}

    tool.active_set = "b"
    tool._pick_instance(_event(shift=True))
    assert not tool.point_sets["a"]
    assert tool.point_sets["b"] == {2, 3}

    tool._pick_instance(_event())
    assert not tool.point_sets["b"]
    assert tool.active_set is None


def test_commit_command_is_atomic_and_undoable(qt_app):
    layer, controller, _viewer, tool = _fixture()
    field = tool._field()
    assert field is not None
    command = SplitBetweenCommitCmd(
        title="test split",
        field_name="instances",
        indices=np.asarray([0, 1, 4, 5], dtype=np.int64),
        old_values=np.asarray([0, 0, 2, 2], dtype=np.int32),
        new_values=np.asarray([7, 7, 8, 8], dtype=np.int32),
        layer_ref=__import__("weakref").ref(layer),
        ctx_ref=__import__("weakref").ref(tool.ctx),
    )

    command.execute()
    assert field.data.reshape(-1)[[0, 1, 4, 5]].tolist() == [7, 7, 8, 8]
    command.undo()
    assert field.data.reshape(-1)[[0, 1, 4, 5]].tolist() == [0, 0, 2, 2]
    assert controller.layer_data_modified.calls == 2


def test_deactivate_clears_preview(qt_app):
    layer, _controller, _viewer, tool = _fixture()
    layer.set_temporary_point_opacity(np.asarray([0, 1]), 0.5)

    tool.deactivate()

    assert layer._temporary_point_opacity is None
