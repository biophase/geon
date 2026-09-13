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
from geon.ui.region_growing_dialog import RegionGrowingDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _layer(name: str, *, selected: bool, with_instance: bool) -> PointCloudLayer:
    data = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    data.add_field(
        name="normals",
        data=np.zeros((4, 3), dtype=np.float32),
        field_type=FieldType.NORMAL,
    )
    if with_instance:
        data.add_field(
            name="instances",
            data=np.asarray([-1, 0, 0, 1], dtype=np.int32),
            field_type=FieldType.INSTANCE,
        )
    layer = PointCloudLayer(data, browser_name=name)
    if selected:
        layer.active_selection = np.asarray([1, 3], dtype=np.int32)
    return layer


def test_dialog_uses_three_tabs_and_restores_output_settings(qt_app) -> None:
    layer = _layer("selected", selected=True, with_instance=True)
    scene = SimpleNamespace(layers={layer.id: layer})

    dialog = RegionGrowingDialog(
        scene,
        layer,
        settings={
            "on_selection_only": True,
            "output_mode": "write_existing",
            "output_existing_field_name": "instances",
        },
    )

    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        "Base",
        "Advanced",
        "Chunking & Merge",
    ]
    assert dialog.on_selection_only() is True
    assert dialog.output_mode() == "write_existing"
    assert dialog.existing_field_name() == "instances"
    assert not dialog.existing_field_combo.isHidden()
    assert dialog.field_name_edit.isHidden()


def test_dialog_updates_selection_and_output_availability_by_layer(qt_app) -> None:
    selected = _layer("selected", selected=True, with_instance=True)
    empty = _layer("empty", selected=False, with_instance=False)
    scene = SimpleNamespace(layers={selected.id: selected, empty.id: empty})
    dialog = RegionGrowingDialog(scene, selected)

    assert dialog.selection_only_box.isEnabled()
    assert dialog.write_existing_field_radio.isEnabled()

    dialog.layer_combo.setCurrentIndex(1)

    assert not dialog.selection_only_box.isEnabled()
    assert not dialog.selection_only_box.isChecked()
    assert not dialog.write_existing_field_radio.isEnabled()
    assert dialog.output_mode() == "create_new"


def test_dialog_legacy_settings_default_to_new_field(qt_app) -> None:
    layer = _layer("cloud", selected=False, with_instance=True)
    scene = SimpleNamespace(layers={layer.id: layer})

    dialog = RegionGrowingDialog(
        scene,
        layer,
        settings={"output_field_base": "legacy_regions"},
    )

    assert dialog.output_mode() == "create_new"
    assert dialog.output_field_base() == "legacy_regions"
