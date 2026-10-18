"""Persist Qt window geometry and dock layouts separately from user preferences."""
from PyQt6.QtCore import QByteArray, QSettings


class WindowLayout:
    VERSION = 1

    def __init__(self, window, settings=None):
        self.window = window
        self.settings = settings if settings is not None else QSettings('geon', 'geon')
        # Capture the built-in arrangement before applying any saved session.
        self.default_geometry = window.saveGeometry()
        self.default_state = window.saveState(self.VERSION)

    def restore(self):
        geometry = self.settings.value('window/geometry')
        state = self.settings.value('window/state')
        if isinstance(geometry, QByteArray):
            self.window.restoreGeometry(geometry)
        if isinstance(state, QByteArray):
            if not self.window.restoreState(state, self.VERSION):
                self.window.restoreState(self.default_state, self.VERSION)

    def save(self):
        self.settings.setValue('window/geometry', self.window.saveGeometry())
        self.settings.setValue('window/state', self.window.saveState(self.VERSION))
        self.settings.sync()

    def reset(self):
        self.window.restoreGeometry(self.default_geometry)
        self.window.restoreState(self.default_state, self.VERSION)
        self.settings.remove('window')
        self.settings.sync()
