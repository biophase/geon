import sys

from geon.ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())