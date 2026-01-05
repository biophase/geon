import sys

from geon.ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication
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

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # set_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())