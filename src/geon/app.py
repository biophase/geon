import sys

from geon.ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from config.theme import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # set_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())