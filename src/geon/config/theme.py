from enum import Enum

from typing import Tuple

DEFAULT_OBJ_COLOR: Tuple[float,float,float] = (0.6,0.6,0.6)
DEFAULT_SEGMENTATION_COLOR: Tuple[int,int,int] = (204, 204, 204)
DEFAULT_RENDERER_BACKGROUND: Tuple[float,float,float] = (.1,.1,.1)
SELECTION_COLOR: Tuple[int, int, int] = (255, 128, 0)
VIEWPORT_TEXT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
REFERENCE_LABEL_TEXT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 1.0)
REFERENCE_LABEL_BACKGROUND_COLOR: Tuple[float, float, float] = (0.0, 0.0, 0.0)
REFERENCE_LABEL_BACKGROUND_OPACITY: float = 0.55


class UIStyle(Enum):
    TYPE_LABEL = "color: rgba(128, 128, 128, 128);"


def set_dark_palette(app):
    # Import Qt lazily so headless test/wheel environments can still import geon.config.theme.
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QPalette
    except ImportError as exc:
        raise RuntimeError(
            "set_dark_palette requires PyQt6 with a functional GUI runtime."
        ) from exc

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

