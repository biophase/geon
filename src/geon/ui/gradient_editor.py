"""Relative gradient handles; field ranges are edited by the owning panel."""
import colorsys
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QPainter, QPolygonF, QImage, QPixmap, QIcon
from PyQt6.QtWidgets import QWidget, QColorDialog, QInputDialog, QMenu
from geon.data.definitions import ChannelMap, sample_map


def swatch(mapping, width=64, height=20, channel=None):
    rgb = sample_map(mapping, np.linspace(0, 1, width))
    if isinstance(mapping, ChannelMap):
        values = rgb
        rgb = np.zeros((width, 3))
        rgb[:, channel if channel is not None else 0] = values
    pixels = np.ascontiguousarray(np.tile((rgb*255).astype(np.uint8)[None, :, :], (height, 1, 1)))
    return QPixmap.fromImage(QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy())


class GradientEditor(QWidget):
    changed = pyqtSignal()
    selectionChanged = pyqtSignal(bool)

    def __init__(self, mapping, channel=None, parent=None):
        super().__init__(parent)
        self.mapping, self.channel = mapping, channel
        self.selected = None
        self.dragging = False
        self.setFixedHeight(40)
        self.setMinimumWidth(100)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def position(self, x):
        return float(np.clip((x-12)/max(1, self.width()-24), 0, 1))

    def hit(self, x):
        distances = np.abs(12+self.mapping.color_positions*(self.width()-24)-x)
        i = int(np.argmin(distances))
        return i if distances[i] < 9 else None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(12, 2, swatch(self.mapping, max(1, self.width()-24), 24, self.channel))
        for i, value in enumerate(self.mapping.color_positions):
            x = 12+value*(self.width()-24)
            painter.setBrush(QColor('#ff9900' if i == self.selected else '#aaaaaa'))
            painter.setPen(QColor('#222222'))
            painter.drawPolygon(QPolygonF([QPointF(x, 26), QPointF(x-6, 38), QPointF(x+6, 38)]))

    def select(self, index):
        self.selected = index
        self.selectionChanged.emit(index is not None and 0 < index < len(self.mapping.color_positions)-1)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.select(self.hit(event.position().x()))
            self.dragging = self.selected is not None

    def mouseMoveEvent(self, event):
        i = self.selected
        if self.dragging and i is not None and 0 < i < len(self.mapping.color_positions)-1:
            p = self.mapping.color_positions
            p[i] = np.clip(self.position(event.position().x()), p[i-1]+1e-6, p[i+1]-1e-6)
            self.changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging and (event.position().y() < -24 or event.position().y() > self.height()+24):
            self.remove_stop()
        self.dragging = False

    def remove_stop(self):
        i = self.selected
        if i is None or i == 0 or i == len(self.mapping.color_positions)-1:
            return
        self.mapping.color_positions = np.delete(self.mapping.color_positions, i)
        key = 'output_values' if isinstance(self.mapping, ChannelMap) else 'color_definitions'
        setattr(self.mapping, key, np.delete(getattr(self.mapping, key), i, axis=0))
        self.select(None)
        self.changed.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_stop()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        i = self.hit(event.position().x())
        if i is not None:
            self.edit_stop(i)
            return
        channel = isinstance(self.mapping, ChannelMap)
        if i is None:
            position = self.position(event.position().x())
            if position <= 0 or position >= 1:
                return
            value = sample_map(self.mapping, np.array([position]))[0]
            if not channel and self.mapping.color_type == 'hsv':
                value = colorsys.rgb_to_hsv(*value)
            i = int(np.searchsorted(self.mapping.color_positions, position))
            self.mapping.color_positions = np.insert(self.mapping.color_positions, i, position)
            key = 'output_values' if channel else 'color_definitions'
            setattr(self.mapping, key, np.insert(getattr(self.mapping, key), i, value, axis=0))
        self.select(i)
        self.changed.emit()

    def contextMenuEvent(self, event):
        index = self.hit(event.pos().x())
        if index is None:
            return
        self.select(index)
        menu = QMenu(self)
        menu.addAction('Edit value', lambda: self.edit_stop(index))
        remove = menu.addAction('Remove', self.remove_stop)
        remove.setEnabled(0 < index < len(self.mapping.color_positions)-1)
        menu.exec(event.globalPos())

    def edit_stop(self, i):
        if isinstance(self.mapping, ChannelMap):
            value, ok = QInputDialog.getDouble(self, 'Channel brightness', 'Intensity:',
                                              float(self.mapping.output_values[i]), 0, 1, 4)
            if not ok:
                return
            self.mapping.output_values[i] = value
        else:
            rgb = sample_map(self.mapping, np.array([self.mapping.color_positions[i]]))[0]
            value = QColorDialog.getColor(QColor.fromRgbF(*rgb), self)
            if not value.isValid():
                return
            rgb = value.getRgbF()[:3]
            self.mapping.color_definitions[i] = colorsys.rgb_to_hsv(*rgb) if self.mapping.color_type == 'hsv' else rgb
        self.select(i)
        self.changed.emit()
