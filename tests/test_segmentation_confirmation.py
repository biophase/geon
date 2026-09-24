from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox

from geon.data.pointcloud import FieldType, PointCloudData
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.connected_components_dialog import ConnectedComponentsDialog
from geon.ui.corner_cleanup_dialog import CornerCleanupDialog
from geon.ui.region_growing_dialog import RegionGrowingDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(params=[
    (ConnectedComponentsDialog, "write_existing"),
    (RegionGrowingDialog, "write_existing"),
    (CornerCleanupDialog, "update_source"),
])
def segmentation_dialog(request, qt_app):
    data = PointCloudData(np.zeros((4, 3), dtype=np.float32))
    data.add_field(name="instances <selected>", data=np.array([0, 0, 1, -1]),
                   field_type=FieldType.INSTANCE)
    layer = PointCloudLayer(data)
    layer.active_selection = np.array([0, 1], dtype=np.int32)
    cls, mode = request.param
    dialog = cls(SimpleNamespace(layers={layer.id: layer}), layer,
                 settings={"output_mode": mode, "on_selection_only": False,
                           "output_existing_field_name": "instances <selected>"})
    yield dialog
    dialog.close()


@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.Yes,
                                     QMessageBox.StandardButton.No, 0])
def test_global_overwrite_requires_confirmation(segmentation_dialog, monkeypatch, answer):
    dialog = segmentation_dialog
    warnings = []

    def respond(warning):
        warnings.append(warning.text())
        assert warning.icon() == QMessageBox.Icon.Warning
        assert warning.textFormat() == Qt.TextFormat.PlainText
        assert warning.standardButtons() == (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        assert warning.standardButton(warning.defaultButton()) == QMessageBox.StandardButton.No
        assert warning.standardButton(warning.escapeButton()) == QMessageBox.StandardButton.No
        return answer

    monkeypatch.setattr(QMessageBox, "exec", respond)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    assert warnings == [
        "You are about to overwrite the field instances <selected> globally. "
        "Are you sure you want to do this? This can't be undone."
    ]
    assert bool(accepted) == (answer == QMessageBox.StandardButton.Yes)
    assert dialog.isVisible() == (answer != QMessageBox.StandardButton.Yes)


@pytest.mark.parametrize("selection_only, create_new", [(True, False), (False, True), (True, True)])
def test_selection_or_new_field_needs_no_warning(segmentation_dialog, monkeypatch,
                                                selection_only, create_new):
    dialog = segmentation_dialog
    dialog.selection_only_box.setChecked(selection_only)
    if create_new:
        radio = getattr(dialog, "create_new_radio", None)
        if radio is None:
            radio = dialog.create_new_field_radio
        radio.setChecked(True)

    def unexpected_warning(_warning):
        pytest.fail("Selection-only and new-field output must not warn about global overwriting")

    monkeypatch.setattr(QMessageBox, "exec", unexpected_warning)
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_cancel_does_not_show_warning(segmentation_dialog, monkeypatch):
    def unexpected_warning(_warning):
        pytest.fail("Cancelling segmentation must not warn")

    monkeypatch.setattr(QMessageBox, "exec", unexpected_warning)
    segmentation_dialog.reject()
    assert segmentation_dialog.result() == QDialog.DialogCode.Rejected
