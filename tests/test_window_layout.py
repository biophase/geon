from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
import pytest

from geon.ui.window_layout import WindowLayout
from geon.ui.preferences_dialog import PreferencesDialog
from geon.settings import Preferences


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def make_window(settings):
    window = QMainWindow()
    window.resize(500, 400)
    window.setCentralWidget(QWidget())
    dock = QDockWidget('Panel', window)
    dock.setObjectName('Panel')
    dock.setWidget(QWidget())
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    layout = WindowLayout(window, settings)
    return window, dock, layout


def test_layout_round_trip_and_reset(app, tmp_path):
    path = str(tmp_path / 'layout.ini')
    window, dock, layout = make_window(QSettings(path, QSettings.Format.IniFormat))
    window.resize(650, 500)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    layout.save()
    restored, panel, saved = make_window(QSettings(path, QSettings.Format.IniFormat))
    saved.restore()
    assert restored.size() == window.size()
    assert restored.dockWidgetArea(panel) == Qt.DockWidgetArea.RightDockWidgetArea
    assert panel.isHidden()
    saved.reset()
    assert restored.width() == 500
    assert restored.height() == 400
    assert restored.dockWidgetArea(panel) == Qt.DockWidgetArea.LeftDockWidgetArea
    assert not panel.isHidden()
    assert not saved.settings.contains('window/state')
    assert not saved.settings.contains('window/geometry')
    window.close()
    restored.close()


def test_reset_preferences_action(app):
    dialog = PreferencesDialog(Preferences())
    requests = []
    dialog.resetWindowArrangementRequested.connect(lambda: requests.append(True))
    dialog.reset_window_arrangement_button.click()
    assert requests == [True]
    dialog.close()
