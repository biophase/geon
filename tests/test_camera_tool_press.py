from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geon.ui.viewer import InteractorStyle


@pytest.mark.parametrize('double_click', [False, True])
def test_tool_click_cannot_start_camera_drag_after_modal_dialog(monkeypatch, double_click):
    monkeypatch.setattr('geon.ui.viewer.time.time', lambda: 100.)
    style = SimpleNamespace(camera_enabled=False, event_info=SimpleNamespace(pos=(10, 20)),
                            last_click_time=99.9 if double_click else 0.,
                            OnLeftButtonDown=Mock(), double_click_event=Mock())
    def accept_dialog(_event):
        # The release can arrive in the modal event loop before this returns.
        style.camera_enabled = True
        style.mode_tool = None
    style.mode_tool = SimpleNamespace(left_button_press_hook=accept_dialog)
    InteractorStyle.left_button_press_event(style, None, None)
    style.OnLeftButtonDown.assert_not_called()
    style.double_click_event.assert_not_called()


def test_normal_camera_press_still_starts_drag():
    style = SimpleNamespace(camera_enabled=True, event_info=SimpleNamespace(pos=(10, 20)),
                            last_click_time=0., mode_tool=None, OnLeftButtonDown=Mock())
    InteractorStyle.left_button_press_event(style, None, None)
    style.OnLeftButtonDown.assert_called_once()
