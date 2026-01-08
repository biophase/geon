import sys

from geon.ui.main_window import MainWindow
from geon.settings import Preferences
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtCore import Qt
from config.theme import *

if sys.platform == 'darwin':
    from PyQt6 import QtCore
    _prev_msg_handler = None
    def _qt_msg_filter(mode, ctx, msg):
        if "QPainter::begin: Paint device returned engine == 0" in msg:
            return  
        if _prev_msg_handler:
            _prev_msg_handler(mode, ctx, msg)

    _prev_msg_handler = QtCore.qInstallMessageHandler(_qt_msg_filter)

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    shints = app.styleHints()
    if shints is not None:
        shints.setColorScheme(Qt.ColorScheme.Dark)
    prefs = Preferences.load()
    win = MainWindow(preferences=prefs)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
