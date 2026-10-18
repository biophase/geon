"""Dockable, draft-based field color editor."""
from copy import deepcopy, copy
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QThreadPool
from PyQt6.QtGui import QIcon, QColor, QPainter, QPalette
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QToolButton, QMenu, QLineEdit, QMessageBox, QDialog,
    QDialogButtonBox, QScrollArea, QSizePolicy, QStyle, QStyleOptionFrame)
from geon.config.theme import UIStyle
from geon.data.definitions import (ColorMap, VectorColorMap, PARAM_MAP, vector_preset,
                                  sample_map, finite_extrema, resolve_limits)
from geon.data.pointcloud import FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.rendering.color_mapping import default_map, components
from geon.io.color_maps import save_map
from geon.util.resources import resource_path
from .gradient_editor import GradientEditor, swatch
from .color_histogram import HistogramJob


class AdditiveBars(pg.BarGraphItem):
    def paint(self, painter, *args):
        painter.save()
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        super().paint(painter, *args)
        painter.restore()


class HistogramAxis(pg.AxisItem):
    def drawPicture(self, painter, axis_spec, tick_specs, text_specs):
        # Keep grid lines even when axis text is hidden, without a bright spine.
        _, start, end = axis_spec
        super().drawPicture(painter, (pg.mkPen(None), start, end), tick_specs, text_specs)


class HistogramPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__(axisItems={name: HistogramAxis(name) for name in ('bottom', 'left')})
        self.setBackground(self.palette().color(QPalette.ColorRole.Window))
        for name in ('bottom', 'left'):
            self.getAxis(name).setPen(self.palette().color(QPalette.ColorRole.Text))
            self.getAxis(name).setTextPen(self.palette().color(QPalette.ColorRole.Text))
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=True, y=True, alpha=0.18)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw the native Qt frame last so plot contents cannot cover its edges.
        option = QStyleOptionFrame()
        option.initFrom(self)
        option.rect = self.viewport().rect()
        option.lineWidth = self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, option, self)
        option.state |= QStyle.StateFlag.State_Sunken
        painter = QPainter(self.viewport())
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Frame, option, painter, self)


class MapCombo(QComboBox):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    def showPopup(self):
        self.panel.refresh_maps()
        super().showPopup()


class ColorChannels(QDockWidget):
    HISTOGRAM_MAX_HEIGHT = 260

    def __init__(self, window):
        super().__init__('Color channels', window)
        self.setObjectName('ColorChannels')
        content_font = self.font()
        content_font.setBold(False)
        title_font = self.font()
        title_font.setBold(True)
        self.setFont(title_font)
        self.window = window
        self.layer = self.field = self.draft = self.original = None
        self.source_field = None
        self.baseline = None
        self.baseline_key = None
        self.preview_key = None
        self.dirty = False
        self.valid = True
        self.generation = 0
        self.hist_dirty = True
        self.hist_data = None
        self.range_mode = 'extrema'
        self.jobs = {}
        self.extrema = []
        self.needs_rebuild = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.refresh_histogram)
        self.visibilityChanged.connect(self.on_visible)
        self.body = QWidget()
        self.body.setFont(content_font)
        self.layout = QVBoxLayout(self.body)
        self.layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(self.gray('Current field: '))
        self.label = QLabel('No active point-cloud field')
        heading.addWidget(self.label, 1)
        self.layout.addLayout(heading)
        self.controls = QWidget()
        self.controls_layout = QVBoxLayout(self.controls)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(4)
        self.controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.layout.addWidget(self.controls)
        self.status = QLabel()
        self.layout.addWidget(self.status)
        settings = self.settings = QToolButton(self.body)
        settings.setIcon(QIcon(resource_path('settings.png')))
        settings.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        settings.setStyleSheet('QToolButton::menu-indicator { image: none; width: 0px; }')
        menu = QMenu(settings)
        for text, mode in [('Show extrema', 'extrema'), ('Show set range', 'range')]:
            menu.addAction(text, lambda checked=False, m=mode: self.set_range_mode(m))
        menu.addSeparator()
        menu.addAction('Rename color map', self.rename)
        settings.setMenu(menu)
        self.controls.setMinimumHeight(max(settings.sizeHint().height(), QComboBox().sizeHint().height()))
        self.plot = HistogramPlot()
        self.plot.setMinimumHeight(180)
        self.plot.setMaximumHeight(self.HISTOGRAM_MAX_HEIGHT)
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.plot.setLabel('bottom', 'Value')
        self.plot.setLabel('left', 'Visible points')
        axis_menu = self.plot.getViewBox().menu.addMenu('Axes')
        self.axis_actions = {}
        for axis_name, name in [('bottom', 'X'), ('left', 'Y')]:
            for kind in ('label', 'numbers'):
                action = axis_menu.addAction(f'{name}-axis {kind}')
                action.setCheckable(True)
                action.setChecked(False)
                self.set_axis_visible(axis_name, kind, False)
                action.toggled.connect(lambda visible, a=axis_name, k=kind: self.set_axis_visible(a, k, visible))
                self.axis_actions[axis_name, kind] = action
        self.layout.addWidget(self.plot)
        self.edit_area = QWidget()
        self.edits = QVBoxLayout(self.edit_area)
        self.edits.setContentsMargins(0, 0, 0, 0)
        self.edits.setSpacing(4)
        self.edit_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.layout.addWidget(self.edit_area)
        self.error = QLabel()
        self.error.setStyleSheet('color: red;')
        self.layout.addWidget(self.error)
        row = QHBoxLayout()
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save)
        self.revert_button = QPushButton('Revert')
        self.revert_button.clicked.connect(self.revert)
        row.addWidget(self.save_button)
        row.addWidget(self.revert_button)
        self.layout.addLayout(row)
        self.layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        self.setWidget(scroll)
        self.rebuild()
        self.hide()

    @property
    def dataset(self):
        return self.window.dataset_manager._dataset

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def gray(self, text):
        label = QLabel(text)
        label.setStyleSheet(UIStyle.TYPE_LABEL.value)
        return label

    def set_axis_visible(self, axis_name, kind, visible):
        axis = self.plot.getAxis(axis_name)
        if kind == 'label':
            axis.showLabel(visible)
        else:
            axis.setStyle(showValues=visible)

    def draft_key(self):
        if self.draft is None:
            return None
        return (self.draft.name, self.draft.signature(), self.field.scalar_component,
                tuple(self.field.rgb_components))

    def track_changes(self):
        try:
            self.dirty = self.draft_key() != self.baseline_key
        except ValueError:
            self.dirty = True
        return self.dirty

    def allow_switch(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Unsaved color map', 'Save or revert the color-map changes?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        if answer == QMessageBox.StandardButton.Discard:
            self.revert()
            return True
        return False

    def activate(self, layer):
        field = layer.active_field if isinstance(layer, PointCloudLayer) else None
        if layer is self.layer and field is self.source_field:
            return True
        if not self.allow_switch():
            if self.layer is not None and self.field is not None:
                self.layer._active_field_name = self.field.name
                self.layer.update()
            return False
        if self.layer is not None:
            if self.field is not None:
                self.layer._color_overrides.pop(self.field.name, None)
            for signal in (self.layer.signals.pointVisibilityChanged, self.layer.signals.fieldDataChanged):
                try:
                    signal.disconnect(self.invalidate)
                except TypeError:
                    pass
            try:
                self.layer.signals.activeFieldChanged.disconnect(self.active_field_changed)
            except TypeError:
                pass
        self.layer = layer if isinstance(layer, PointCloudLayer) else None
        self.source_field = field
        self.field = copy(field) if field is not None else None
        field = self.field
        self.hist_data = None
        if field is None:
            self.draft = None
            self.label.setText('No active point-cloud field')
        else:
            self.baseline = (deepcopy(field.color_map), field.scalar_component, field.rgb_components)
            self.original = deepcopy(field.color_map)
            self.draft = deepcopy(default_map(field)) if field.field_type not in (FieldType.SEMANTIC, FieldType.INSTANCE) else None
            if self.draft is not None:
                field.color_map = deepcopy(self.draft)
                layer._color_overrides[field.name] = field
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            self.label.setText(f'{layer.id}/{field.name}')
            layer.signals.pointVisibilityChanged.connect(self.invalidate)
            layer.signals.fieldDataChanged.connect(self.invalidate)
            layer.signals.activeFieldChanged.connect(self.active_field_changed)
        self.dirty = False
        self.baseline_key = self.preview_key = self.draft_key()
        self.rebuild()
        self.invalidate()
        return True

    def active_field_changed(self):
        self.activate(self.layer)

    def channel_combo(self, component, callback):
        combo = QComboBox()
        combo.setEditable(True)
        count = 1 if self.field.data.ndim == 1 else self.field.data.shape[1]
        try:
            names = FieldType.get_channel_names(self.field.field_type)
        except KeyError:
            names = ()
        for i in range(count):
            combo.addItem(names[i] if i < len(names) else str(i), i)
        combo.setCurrentIndex(component)
        last = [component]
        def selected(index):
            value = combo.itemData(index)
            if value != last[0]:
                last[0] = value
                callback(value)
        combo.activated.connect(selected)
        def typed():
            text = combo.currentText()
            index = combo.findText(text)
            if index < 0:
                try:
                    index = int(text)
                except ValueError:
                    index = -1
            if 0 <= index < count:
                combo.setCurrentIndex(index)
                if index != last[0]:
                    last[0] = index
                    callback(index)
            else:
                combo.setCurrentIndex(last[0])
        combo.lineEdit().editingFinished.connect(typed)
        return combo

    def rebuild(self):
        # Preserve the shared settings button while rebuilding channel controls.
        self.settings.setParent(self.body)
        self.clear_layout(self.controls_layout)
        self.clear_layout(self.edits)
        self.valid = True
        self.error.clear()
        self.save_button.setEnabled(self.draft is not None)
        self.revert_button.setEnabled(self.draft is not None)
        channel_row = QHBoxLayout()
        self.controls_layout.addLayout(channel_row)
        if self.draft is None:
            channel_row.addStretch(1)
            channel_row.addWidget(self.settings)
            self.settings.show()
            self.plot.clear()
            self.status.setText('Instance color editing is not available yet' if self.field and self.field.field_type == FieldType.INSTANCE else '')
            return
        vector = isinstance(self.draft, VectorColorMap)
        count = 1 if self.field.data.ndim == 1 else self.field.data.shape[1]
        if vector:
            row = channel_row
            row.addWidget(self.gray('Channel map: '))
            for i, text in enumerate(('Red', 'Green', 'Blue')):
                row.addWidget(self.channel_combo(self.field.rgb_components[i], lambda v, c=i: self.bind(c, v)))
                row.addWidget(QLabel(': '+text))
        elif count > 1:
            row = channel_row
            row.addWidget(self.gray('Input channel: '))
            row.addWidget(self.channel_combo(self.field.scalar_component, lambda v: self.bind(None, v)))
        channel_row.addStretch(1)
        channel_row.addWidget(self.settings)
        self.settings.show()
        row = QHBoxLayout()
        row.addWidget(self.gray('Color map: '))
        self.combo = MapCombo(self)
        self.refresh_maps()
        self.combo.activated.connect(self.choose_map)
        row.addWidget(self.combo, 1)
        duplicate = QPushButton('Duplicate')
        duplicate.clicked.connect(self.duplicate)
        row.addWidget(duplicate)
        rename = QPushButton('Rename')
        rename.clicked.connect(self.rename)
        row.addWidget(rename)
        self.edits.addLayout(row)
        maps = self.draft.channels if vector else (self.draft,)
        state = self.layer._color_mapping.states.get('Foreground') if self.layer._color_mapping else None
        self.extrema = state[1] if state and state[0].name == self.field.name and len(state[1]) == (3 if vector else 1) else [finite_extrema(v) for v in components(self.field, self.draft)]
        self.limit_inputs = []
        for i, mapping in enumerate(maps):
            channel_layout = QVBoxLayout()
            channel_layout.setSpacing(0)
            self.edits.addLayout(channel_layout)
            if vector:
                channel_layout.addWidget(QLabel(('Red', 'Green', 'Blue')[i]))
            editor = GradientEditor(mapping, i if vector else None)
            row = QHBoxLayout()
            editor.changed.connect(self.changed)
            for key, value in zip(('min', 'max'), self.extrema[i]):
                if key == 'max':
                    row.addWidget(editor, 1)
                row.addWidget(self.gray(key.capitalize()+': '))
                box = QLineEdit('' if getattr(mapping, key) is None else str(getattr(mapping, key)))
                box.setFixedWidth(66)
                box.setPlaceholderText(f'{value:g}')
                box.editingFinished.connect(self.limits_changed)
                self.limit_inputs.append((box, mapping, key))
                row.addWidget(box)
            channel_layout.addLayout(row)

    def refresh_maps(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        maps = list(self.dataset.unique_color_maps) if self.dataset else []
        maps += [deepcopy(m) for m in PARAM_MAP.values()]
        maps += [vector_preset(-1, 1), vector_preset(0, 1), vector_preset(0, 255)]
        if self.draft is not None and self.valid:
            maps.insert(0, deepcopy(self.draft))
        seen = set()
        count = 1 if self.field.data.ndim == 1 else self.field.data.shape[1]
        for mapping in maps:
            key = (mapping.name, mapping.signature())
            if key in seen or (isinstance(mapping, VectorColorMap) and count < 3):
                continue
            seen.add(key)
            icon = QIcon(resource_path('rgb.png')) if isinstance(mapping, VectorColorMap) else QIcon(swatch(mapping, 24, 24))
            self.combo.addItem(icon, mapping.name, mapping)
        self.combo.blockSignals(False)

    def choose_map(self, index):
        mapping = deepcopy(self.combo.itemData(index))
        if self.draft is not None and self.valid and (mapping.name, mapping.signature()) == (self.draft.name, self.draft.signature()):
            return
        if not self.allow_switch():
            self.combo.setCurrentIndex(0)
            return
        try:
            channel_maps = mapping.channels if isinstance(mapping, VectorColorMap) else (mapping,)
            for channel_map, values in zip(channel_maps, components(self.field, mapping)):
                resolve_limits(channel_map, finite_extrema(values))
        except ValueError as exc:
            self.error.setText(str(exc))
            self.combo.setCurrentIndex(0)
            return
        available = self.dataset.unique_color_maps if self.dataset else []
        existing = any(m.name == mapping.name and m.signature() == mapping.signature() for m in available)
        self.original = deepcopy(mapping) if existing else None
        names = {m.name for m in available}
        if not existing and mapping.name in names:
            base, suffix = mapping.name, 1
            while f'{base}.{suffix:03}' in names:
                suffix += 1
            mapping.name = f'{base}.{suffix:03}'
        self.draft = mapping
        self.field.color_map = self.draft
        count = self.field.data.shape[1] if self.field.data.ndim > 1 else 1
        if any(v >= count for v in self.field.rgb_components):
            self.field.rgb_components = (0, 1, 2)
        self.track_changes()
        self.layer.update()
        self.rebuild()
        self.preview_key = self.draft_key()
        self.invalidate()
        self.window.viewer.rerender()

    def bind(self, channel, value):
        current = self.field.scalar_component if channel is None else self.field.rgb_components[channel]
        if current == value:
            return
        if channel is None:
            self.field.scalar_component = value
        else:
            values = list(self.field.rgb_components)
            values[channel] = value
            self.field.rgb_components = tuple(values)
        self.field.color_map = self.draft
        self.track_changes()
        self.layer.update()
        self.rebuild()
        self.preview_key = self.draft_key()
        self.invalidate()
        self.window.viewer.rerender()

    def changed(self):
        self.track_changes()
        try:
            self.draft.validate()
            key = self.draft_key()
            if self.valid and key == self.preview_key:
                return
            maps = self.draft.channels if isinstance(self.draft, VectorColorMap) else (self.draft,)
            for mapping, extrema in zip(maps, self.extrema):
                resolve_limits(mapping, extrema)
            self.field.color_map = deepcopy(self.draft)
            self.layer.preview_color_map(self.field, self.draft)
            self.error.clear()
            self.valid = True
            self.preview_key = key
            self.window.viewer.rerender()
            self.draw_histogram()
        except ValueError as exc:
            self.error.setText(str(exc))
            self.valid = False
        self.save_button.setEnabled(self.valid)

    def limits_changed(self):
        try:
            parsed = [(mapping, key, float(box.text()) if box.text().strip() else None)
                      for box, mapping, key in self.limit_inputs]
            if self.valid and all(getattr(mapping, key) == value for mapping, key, value in parsed):
                return
            for mapping, key, value in parsed:
                setattr(mapping, key, value)
            self.changed()
            if self.range_mode == 'range':
                self.invalidate()
        except ValueError:
            self.dirty = True
            self.error.setText('Enter a finite number or leave the limit empty')
            self.valid = False
            self.save_button.setEnabled(False)

    def rename(self):
        if self.draft is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Rename color map')
        layout = QVBoxLayout(dialog)
        box = QLineEdit(self.draft.name)
        layout.addWidget(box)
        error = QLabel('Name already taken')
        error.setStyleSheet('color: red;')
        layout.addWidget(error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        names = {m.name for m in self.dataset.unique_color_maps} if self.dataset else set()
        def validate():
            taken = box.text().strip() in names and box.text().strip() != self.draft.name
            error.setVisible(taken)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(box.text().strip()) and not taken)
        box.textChanged.connect(validate)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        validate()
        if dialog.exec():
            self.draft.name = box.text().strip()
            self.track_changes()
            self.refresh_maps()

    def duplicate(self):
        names = {m.name for m in self.dataset.unique_color_maps} if self.dataset else set()
        name = self.draft.name
        index = 1
        while f'{name}.{index:03}' in names:
            index += 1
        self.draft = deepcopy(self.draft)
        self.draft.name = f'{name}.{index:03}'
        self.original = None
        self.track_changes()
        self.rebuild()

    def save(self):
        if self.draft is None or not self.valid:
            return False
        if not self.dirty:
            return True
        try:
            if self.dataset:
                old_bindings = self.source_field.scalar_component, self.source_field.rgb_components
                self.source_field.scalar_component, self.source_field.rgb_components = self.field.scalar_component, self.field.rgb_components
                try:
                    save_map(self.dataset, self.original, self.draft, self.layer.data, self.source_field)
                except Exception:
                    self.source_field.scalar_component, self.source_field.rgb_components = old_bindings
                    raise
            else:
                self.source_field.color_map = deepcopy(self.draft)
                self.source_field.scalar_component, self.source_field.rgb_components = self.field.scalar_component, self.field.rgb_components
            self.original = deepcopy(self.draft)
            self.baseline = (deepcopy(self.draft), self.field.scalar_component, self.field.rgb_components)
            self.dirty = False
            self.baseline_key = self.preview_key = self.draft_key()
            self.layer.update()
            scene_manager = getattr(self.window, 'scene_manager', None)
            scene = getattr(scene_manager, '_scene', None)
            if scene is not None:
                for other in scene._layers.values():
                    if other is not self.layer and isinstance(other, PointCloudLayer):
                        other.update()
            self.window.viewer.rerender()
            return True
        except Exception as exc:
            QMessageBox.critical(self, 'Could not save color map', str(exc))
            return False

    def revert(self):
        if self.field is None or self.baseline is None:
            return
        self.field.color_map, self.field.scalar_component, self.field.rgb_components = deepcopy(self.baseline)
        self.original = deepcopy(self.field.color_map)
        self.draft = deepcopy(default_map(self.field))
        self.dirty = False
        self.baseline_key = self.preview_key = self.draft_key()
        self.layer.update()
        self.rebuild()
        self.invalidate()
        self.window.viewer.rerender()

    def set_range_mode(self, mode):
        self.range_mode = mode
        self.invalidate()

    def on_visible(self, visible):
        if visible and self.hist_dirty:
            self.timer.start()
        elif not visible:
            self.timer.stop()
            self.generation += 1
            self.hist_dirty = True

    def invalidate(self):
        if self.field is not None and self.source_field is not None:
            self.field.data = self.source_field.data
            state = self.layer._color_mapping.states.get('Foreground') if self.layer._color_mapping else None
            if state and state[0].name == self.field.name:
                self.extrema = state[1]
        self.hist_dirty = True
        self.generation += 1
        if self.isVisible():
            self.timer.start()

    def refresh_histogram(self):
        if not self.isVisible() or self.field is None or self.field.field_type == FieldType.INSTANCE:
            return
        if self.jobs:
            return
        semantic = self.field.field_type == FieldType.SEMANTIC
        values = [self.field.data.reshape(-1)] if semantic else components(self.field, self.draft)
        limits = None
        if not semantic and self.range_mode == 'range':
            try:
                maps = self.draft.channels if isinstance(self.draft, VectorColorMap) else (self.draft,)
                ranges = [resolve_limits(m, finite_extrema(v)) for m, v in zip(maps, values)]
                limits = min(v[0] for v in ranges), max(v[1] for v in ranges)
            except ValueError as exc:
                self.error.setText(str(exc))
                return
        job = HistogramJob(self.generation, values, self.layer._visibility_mask, limits, semantic)
        self.jobs[self.generation] = job
        job.result.ready.connect(self.histogram_ready)
        job.result.failed.connect(self.histogram_failed)
        self.status.setText('Calculating…')
        QThreadPool.globalInstance().start(job)

    def histogram_ready(self, generation, edges, counts):
        self.jobs.pop(generation, None)
        if generation != self.generation or not self.isVisible():
            if self.isVisible() and self.hist_dirty:
                self.timer.start()
            return
        self.hist_dirty = False
        self.hist_data = edges, counts
        self.status.setText('' if len(edges) else 'No visible finite values')
        self.draw_histogram()
        if len(edges):
            self.plot.autoRange()

    def histogram_failed(self, generation, message):
        self.jobs.pop(generation, None)
        if generation == self.generation:
            self.status.setText(message)
        elif self.isVisible() and self.hist_dirty:
            self.timer.start()

    def draw_histogram(self):
        self.plot.clear()
        if not self.valid:
            return
        if self.hist_data is None or self.field is None:
            return
        edges, counts = self.hist_data
        if not len(edges):
            return
        if self.field.field_type == FieldType.SEMANTIC:
            colors = self.field.schema.get_color_array(edges.astype(np.int32))[:, :3]
            colors[edges == -1] = self.layer.unassigned_point_color[:3]
            x = np.arange(len(edges))
            self.plot.addItem(pg.BarGraphItem(x=x, height=counts[0], width=.8, brushes=[QColor(*map(int,c)) for c in colors], pen=None))
            self.plot.getAxis('bottom').setTicks([[(i, str(int(v))) for i,v in enumerate(edges)]])
            return
        if self.draft is None:
            return
        self.plot.getAxis('bottom').setTicks(None)
        centers = (edges[:-1]+edges[1:])/2
        vector = isinstance(self.draft, VectorColorMap)
        maps = self.draft.channels if vector else (self.draft,)
        for i, (mapping, heights) in enumerate(zip(maps, counts)):
            low, high = resolve_limits(mapping, self.extrema[i])
            t = np.full(len(centers), .5) if low == high else (centers-low)/(high-low)
            values = sample_map(mapping, t)
            if vector:
                rgb = np.zeros((len(t),3))
                rgb[:,i] = 1.0
            else:
                rgb = values
            brushes = [QColor(*map(int, c*255)) for c in rgb]
            cls = AdditiveBars if vector else pg.BarGraphItem
            self.plot.addItem(cls(x=centers, height=heights, width=np.diff(edges), brushes=brushes, pen=None))
