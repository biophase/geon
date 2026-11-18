"""Asynchronous reconstruction with a reversible, disposable preview."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar
import weakref

import numpy as np
import vtk
from vtk.util import numpy_support as ns
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton, QComboBox,
                             QSlider, QCheckBox, QMessageBox, QDialog, QVBoxLayout,
                             QPlainTextEdit, QDialogButtonBox)
from geon.data.pointcloud import PointCloudData, FieldBase, FieldType
from geon.data.definitions import ColorMap
from geon.data.mesh import MeshLevel
from geon.rendering.mesh import MeshLayer
from geon.rendering.pointcloud import PointCloudLayer
from .base import ModeTool
from .command_manager import Command
from geon.algorithms.reconstruction.timing import ProgressTimer


def _refresh_scene(ctx):
    ctx.controller.scene_tree_request_change.emit()
    manager = getattr(ctx.viewer.window(), "scene_manager", None)
    if manager is not None and ctx.scene.active_layer is not None:
        manager.broadcastActivatedLayer.emit(ctx.scene.active_layer)
    ctx.viewer.rerender()


@dataclass
class AddReconstructedMeshCmd(Command):
    layer: MeshLayer
    scene_ref: object
    ctx_ref: object
    previous_active: str | None = None

    def execute(self):
        scene, ctx = self.scene_ref(), self.ctx_ref()
        if scene is None or ctx is None:
            return
        if self.layer.id in scene.doc.scene_items or self.layer.id in scene.layers:
            raise ValueError("The reconstructed mesh ID is already in use.")
        scene.doc.add_data(self.layer.data)
        try:
            scene.add_layer(self.layer)
        except Exception:
            scene.doc.remove_data(self.layer.id)
            raise
        scene.active_layer_id = self.layer.id
        _refresh_scene(ctx)
        ctx.controller.layer_data_modified.emit(self.layer)

    def undo(self):
        scene, ctx = self.scene_ref(), self.ctx_ref()
        if scene is None or ctx is None:
            return
        scene.remove_layer(self.layer.id)
        if self.previous_active in scene.layers:
            scene.active_layer_id = self.previous_active
        _refresh_scene(ctx)
        ctx.controller.layer_data_modified.emit(self.layer)


class _ReconstructionWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, cloud, primitive_name, mapped_names, config, mask, spacing, parent=None):
        super().__init__(parent)
        self.cloud, self.primitive_name, self.mapped_names = cloud, primitive_name, mapped_names
        self.config, self.mask, self.spacing = config, mask, spacing

    def run(self):
        from geon.algorithms.reconstruction.transfer import reconstruct_cloud, corner_curves, deviation_surface
        from geon.algorithms.reconstruction.preview import spline_preview
        timer=ProgressTimer()
        def progress(message):
            if self.isInterruptionRequested():
                raise InterruptedError()
            self.status.emit(timer.format(message))
        try:
            mesh, result = reconstruct_cloud(self.cloud, self.primitive_name, self.mapped_names,
                config=self.config, mask=self.mask, progress=progress)
            progress("Preparing border and deviation previews")
            curves = corner_curves(result)
            surface = deviation_surface(result, self.cloud.points,
                self.cloud.get_fields(names=self.primitive_name)[0].data.reshape(-1),
                spacing=self.spacing, mask=self.mask, progress=progress)
            splines=spline_preview(result,self.config.curve_tolerance,progress)
            progress("Preview ready")
            self.ready.emit((mesh, curves, surface, result.point_deviations, splines))
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(timer.format(f'Reconstruction failed: {exc}'))
        finally:
            self.cloud = None
            self.mask = None


def _line_actor(curves, color):
    points, cells = vtk.vtkPoints(), vtk.vtkCellArray()
    offset = 0
    for curve in curves:
        curve = np.asarray(curve)
        for point in curve:
            points.InsertNextPoint(*point)
        cells.InsertNextCell(len(curve), list(range(offset, offset+len(curve))))
        offset += len(curve)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(4)
    actor.GetProperty().LightingOff()
    return actor


def _control_point_actor(points):
    poly=vtk.vtkPolyData()
    xyz=vtk.vtkPoints()
    xyz.SetData(ns.numpy_to_vtk(np.asarray(points,dtype=float),deep=True))
    poly.SetPoints(xyz)
    cells=vtk.vtkCellArray()
    for index in range(len(points)):
        cells.InsertNextCell(1,[index])
    poly.SetVerts(cells)
    mapper=vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor=vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    actor.GetProperty().SetColor(1.,.55,.05)
    actor.GetProperty().SetPointSize(7)
    actor.GetProperty().RenderPointsAsSpheresOn()
    actor.GetProperty().LightingOff()
    return actor


@dataclass
class ReconstructMeshTool(ModeTool):
    label: ClassVar = "reconstruct_mesh"
    tooltip: ClassVar = "Mesh from PCD instance field"
    shortcut: ClassVar = None
    icon_path: ClassVar = None
    ui_zones: ClassVar = set()
    use_local_cm: ClassVar = False
    show_in_toolbar: ClassVar = False
    keep_focus: ClassVar = False
    _worker: object = field(default=None, init=False, repr=False)
    preview_layer: object = field(default=None, init=False, repr=False)
    _overlays: list = field(default_factory=list, init=False, repr=False)
    _spline_grid_actors: list = field(default_factory=list, init=False, repr=False)
    _spline_control_actors: list = field(default_factory=list, init=False, repr=False)
    _surface: object = field(default=None, init=False, repr=False)
    _deviation_field: object = field(default=None, init=False, repr=False)
    _deviation_actor: object = field(default=None, init=False, repr=False)
    _status: str = field(default="Starting reconstruction…", init=False)
    _cancelled: bool = field(default=False, init=False)
    _cleaned: bool = field(default=False, init=False)
    _log_messages: list = field(default_factory=list, init=False, repr=False)
    _log_dialog: object = field(default=None, init=False, repr=False)
    _log_timer: object = field(default_factory=ProgressTimer, init=False, repr=False)

    def activate(self):
        super().activate()
        from geon.ui.reconstruction_dialog import ReconstructionDialog
        source = self.ctx.scene.active_layer
        if not isinstance(source, PointCloudLayer):
            QMessageBox.warning(self.ctx.viewer, "Reconstruction", "Activate a point cloud first.")
            self.ctx.controller.deactivate_tool()
            return
        if not source.data.get_fields(field_type=FieldType.INSTANCE):
            QMessageBox.warning(self.ctx.viewer, "Reconstruction", "The point cloud needs an instance field for primitive segmentation.")
            self.ctx.controller.deactivate_tool()
            return
        panel = getattr(self.ctx.viewer.window(), "color_channels", None)
        if panel is not None and not panel.allow_switch():
            self.ctx.controller.deactivate_tool()
            return
        self.source = source
        self.previous_active = self.ctx.scene.active_layer_id
        self.source_visible = source.visible
        dialog = ReconstructionDialog(source, self.ctx.viewer)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.ctx.controller.deactivate_tool()
            return
        self.mesh_name = dialog.name_input.text().strip()
        self._log_messages.clear()
        self._log_timer=ProgressTimer()
        # Freeze only geometry and requested fields. Tools may be switched while
        # the worker cooperatively cancels, without racing mutable source arrays.
        names = list(dict.fromkeys([dialog.primitive_combo.currentData()] + dialog.mapped_names()))
        snapshot = PointCloudData(source.data.points.copy())
        snapshot._fields = [deepcopy(f) for f in source.data.get_fields(names=names)]
        mask = None
        if dialog.selection_only.isChecked():
            mask = np.zeros(len(snapshot.points), dtype=bool)
            mask[source.active_selection] = True
        worker = _ReconstructionWorker(snapshot, names[0], dialog.mapped_names(), dialog.config,
                                       mask, dialog.preview_spacing.value(), self.ctx.controller)
        self._worker = worker
        worker.status.connect(self._set_status)
        worker.ready.connect(self._ready)
        worker.failed.connect(self._failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _worker_finished(self):
        self._worker = None

    def _set_status(self, message):
        message=self._log_timer.format(message)
        self._status = message
        self._log_messages.append(message)
        if self._log_dialog is not None and self._log_dialog.isVisible():
            self._log_dialog.findChild(QPlainTextEdit).appendPlainText(message)
        if hasattr(self, "status_label") and self._in_use:
            self.status_label.setToolTip(message)
            self.status_label.setText(self.status_label.fontMetrics().elidedText(
                ProgressTimer.message(message),Qt.TextElideMode.ElideRight,260))

    def _failed(self, message):
        if self._cancelled or not self._in_use:
            return
        self._set_status(message if message.startswith('[elapsed ') else f'Reconstruction failed: {message}')
        error = QMessageBox(QMessageBox.Icon.Warning, "Reconstruction failed", message,
                           QMessageBox.StandardButton.Ok, self.ctx.viewer)
        error.setDetailedText('\n'.join(self._log_messages))
        error.exec()
        self.ctx.controller.deactivate_tool()

    def _show_log(self):
        if self._log_dialog is None:
            dialog = QDialog(self.ctx.viewer)
            dialog.setWindowTitle('Reconstruction log')
            dialog.resize(850, 480)
            layout = QVBoxLayout(dialog)
            text = QPlainTextEdit(dialog)
            text.setReadOnly(True)
            layout.addWidget(text)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
            buttons.rejected.connect(dialog.close)
            layout.addWidget(buttons)
            dialog.destroyed.connect(lambda: setattr(self, '_log_dialog', None))
            self._log_dialog = dialog
        self._log_dialog.findChild(QPlainTextEdit).setPlainText('\n'.join(self._log_messages))
        self._log_dialog.show()
        self._log_dialog.raise_()

    def create_context_widget(self, parent):
        widget = QWidget(parent)
        layout = QGridLayout(widget)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.status_label = QLabel()
        self.status_label.setFixedWidth(260)
        self.status_label.setToolTip(self._status)
        self.status_label.setText(self.status_label.fontMetrics().elidedText(
            ProgressTimer.message(self._status),Qt.TextElideMode.ElideRight,260))
        layout.addWidget(self.status_label, 0, 0)
        self.log_button = QPushButton('Log')
        self.log_button.clicked.connect(self._show_log)
        layout.addWidget(self.log_button, 1, 4)
        self.corners_button = QPushButton("Borders")
        self.corners_button.setCheckable(True)
        self.corners_button.setChecked(True)
        self.corners_button.setToolTip("Teal: analytic intersections · Green: coplanar joins · Red: outer borders")
        self.corners_button.toggled.connect(self._toggle_corners)
        layout.addWidget(self.corners_button, 0, 1)
        self.field_combo = QComboBox()
        self.field_combo.setFixedWidth(260)
        self.field_combo.setToolTip('Preview colour field')
        self.field_combo.currentIndexChanged.connect(self._field_changed)
        layout.addWidget(self.field_combo, 1, 0)
        layout.addWidget(QLabel("Transparency"), 0, 3)
        self.transparency = QSlider(Qt.Orientation.Horizontal)
        self.transparency.setRange(0, 100)
        self.transparency.setValue(0)
        self.transparency.setFixedWidth(90)
        self.transparency.valueChanged.connect(self._opacity)
        layout.addWidget(self.transparency, 1, 3)
        self.accept_button = QPushButton("Accept")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self._accept)
        layout.addWidget(self.accept_button, 0, 5)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.ctx.controller.deactivate_tool)
        layout.addWidget(cancel, 1, 5)
        self.source_button = QCheckBox("Source")
        self.source_button.toggled.connect(lambda visible: self._source_visibility(visible))
        layout.addWidget(self.source_button, 1, 1)
        self.spline_grid_button=QCheckBox('B-spline UV grid')
        self.spline_grid_button.setChecked(True)
        self.spline_grid_button.setEnabled(False)
        self.spline_grid_button.toggled.connect(self._toggle_spline_overlays)
        layout.addWidget(self.spline_grid_button,0,2)
        self.spline_control_button=QCheckBox('Control points / net')
        self.spline_control_button.setEnabled(False)
        self.spline_control_button.toggled.connect(self._toggle_spline_overlays)
        layout.addWidget(self.spline_control_button,1,2)
        self.spline_count_label=QLabel('')
        layout.addWidget(self.spline_count_label,0,4)
        self.spline_count_label.setMinimumWidth(100)
        # Match Active Selection: exactly two 18-pixel control rows.
        for index in range(layout.count()):
            control=layout.itemAt(index).widget()
            if control is not None:
                control.setFixedHeight(18)
        return widget

    def _ready(self, payload):
        if self._cancelled or not self._in_use:
            return
        mesh, curves, self._surface, distances = payload[:4]
        splines=payload[4] if len(payload)>4 else {}
        # Some final topology diagnostics are stored without a progress signal.
        logged={ProgressTimer.message(entry) for entry in self._log_messages}
        for message in mesh.metadata.get('diagnostics', []):
            if message not in logged:
                self._set_status(f'Final diagnostic: {message}')
        mesh.name = self.mesh_name
        while mesh.id in self.ctx.scene.layers or mesh.id in self.ctx.scene.doc.scene_items:
            mesh.id = mesh._generate_id()
        layer = MeshLayer(mesh)
        layer.is_preview = True
        self.preview_layer = layer
        self.ctx.scene.add_layer(layer)
        layer.set_active_field(MeshLevel.FACE, mesh.metadata["primitive_field"])
        self.ctx.scene.active_layer_id = layer.id
        self.source.set_visible(self.source_button.isChecked())
        colors = {"analytic": (0., .7, .7), "coplanar": (.1, .85, .2), "outer": (.95, .1, .1)}
        for kind, lines in curves.items():
            actor = _line_actor(lines, colors[kind])
            actor.SetVisibility(self.corners_button.isChecked())
            self.ctx.viewer._renderer.AddActor(actor)
            self._overlays.append(actor)
        for overlay in splines.values():
            grid=_line_actor(overlay['grid'],(.75,.15,.9))
            grid.GetProperty().SetLineWidth(1.5)
            grid.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
            grid.GetMapper().SetRelativeCoincidentTopologyLineOffsetParameters(-1.,-1.)
            net=_line_actor(overlay['control_net'],(1.,.55,.05))
            net.GetProperty().SetLineWidth(1.5)
            points=_control_point_actor(overlay['control_points'])
            self._spline_grid_actors.append(grid)
            self._spline_control_actors.extend([net,points])
            for actor in (grid,net,points):
                self.ctx.viewer._renderer.AddActor(actor)
        self.spline_grid_button.setEnabled(bool(splines))
        self.spline_control_button.setEnabled(bool(splines))
        self.spline_count_label.setText(f'{len(splines)} B-spline faces' if splines else 'No B-spline faces')
        self._toggle_spline_overlays()
        mapping = ColorMap("Mesh deviation", color_positions=np.array([0., .35, .7, 1.]),
                           color_definitions=np.array([[.1,.25,.9], [0.,.8,.65], [1.,.9,0.], [.9,.05,.05]]),
                           min=0., max=max(float(np.quantile(distances, .99)), 1e-6))
        self._deviation_field = FieldBase("Mesh deviation (source points)", distances[:, None], FieldType.SCALAR, mapping)
        layer.preview_map_changed = self._preview_map
        self.field_combo.blockSignals(True)
        for field in mesh.get_fields(level=MeshLevel.FACE):
            self.field_combo.addItem(field.name, field.name)
        self.field_combo.addItem("Deviation (temporary)", "__deviation__")
        self.field_combo.blockSignals(False)
        self.accept_button.setEnabled(True)
        self._opacity()
        self._set_status(f"{len(mesh.vertices):,} vertices · {len(mesh.faces):,} faces · Preview")
        _refresh_scene(self.ctx)
        self._activate_color_channels()
        self.ctx.viewer.rerender()

    def _activate_color_channels(self):
        panel = getattr(self.ctx.viewer.window(), "color_channels", None)
        if panel is not None:
            panel.activate(self.preview_layer)
            if self.preview_layer is not None and self.preview_layer.preview_field is not None:
                panel.show()

    def _field_changed(self):
        layer = self.preview_layer
        if layer is None:
            return
        panel = getattr(self.ctx.viewer.window(), "color_channels", None)
        if panel is not None and not panel.allow_switch():
            self.field_combo.blockSignals(True)
            self.field_combo.setCurrentIndex(self.field_combo.findData("__deviation__" if layer.preview_field else layer.active_field_name))
            self.field_combo.blockSignals(False)
            return
        name = self.field_combo.currentData()
        if name == "__deviation__":
            if self._surface is None:
                QMessageBox.information(self.ctx.viewer, "Deviation preview", "The mesh exceeds the temporary deviation surface budget (80,000 vertices).")
                return
            layer.preview_field = self._deviation_field
            self._show_deviation()
            layer.signals.activeFieldChanged.emit()
        else:
            if self._deviation_actor is not None:
                self._deviation_actor.SetVisibility(False)
            layer.set_active_field(MeshLevel.FACE, name)
        self._activate_color_channels()
        self.ctx.viewer.rerender()

    def _show_deviation(self):
        if self._deviation_actor is None:
            vertices, faces, values = self._surface
            self._deviation_poly = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            points.SetData(ns.numpy_to_vtk(vertices, deep=True))
            self._deviation_poly.SetPoints(points)
            cells = vtk.vtkCellArray()
            offsets = np.arange(0, 3*len(faces)+1, 3, dtype=np.int64)
            cells.SetData(ns.numpy_to_vtkIdTypeArray(offsets, deep=True),
                          ns.numpy_to_vtkIdTypeArray(faces.reshape(-1), deep=True))
            self._deviation_poly.SetPolys(cells)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self._deviation_poly)
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1., -1.)
            self._deviation_actor = vtk.vtkActor()
            self._deviation_actor.SetMapper(mapper)
            self._deviation_actor.SetPickable(False)
            self._deviation_actor.GetProperty().LightingOff()
            self.ctx.viewer._renderer.AddActor(self._deviation_actor)
        self._deviation_actor.SetVisibility(True)
        self.preview_layer.update()
        self._opacity()

    def _preview_map(self, field):
        if self._deviation_actor is None or self._surface is None:
            return
        from copy import copy
        sample_field = copy(field)
        sample_field.data = self._surface[2][:, None]
        colors = MeshLayer._colors(sample_field)
        self._deviation_poly.GetPointData().SetScalars(ns.numpy_to_vtk(colors, deep=True))
        self._deviation_poly.Modified()

    def _toggle_corners(self, visible):
        for actor in self._overlays:
            actor.SetVisibility(visible)
        self.ctx.viewer.rerender()

    def _toggle_spline_overlays(self, *_):
        for actor in self._spline_grid_actors:
            actor.SetVisibility(self.spline_grid_button.isChecked())
        for actor in self._spline_control_actors:
            actor.SetVisibility(self.spline_control_button.isChecked())
        self.ctx.viewer.rerender()

    def _source_visibility(self, visible):
        if hasattr(self, "source"):
            self.source.set_visible(visible)
            self.ctx.viewer.rerender()

    def _opacity(self):
        if self.preview_layer is not None:
            opacity = 1-self.transparency.value()/100.
            self.preview_layer.set_opacity(opacity)
            if self._deviation_actor is not None:
                self._deviation_actor.GetProperty().SetOpacity(opacity)
            self.ctx.viewer.rerender()

    def _accept(self):
        if self.preview_layer is None:
            return
        layer = self.preview_layer
        self._cleanup(accept=True)
        layer.is_preview = False
        layer.set_opacity(1.)
        self.ctx.controller.do_global(AddReconstructedMeshCmd("Reconstruct mesh", layer,
            weakref.ref(self.ctx.scene), weakref.ref(self.ctx), self.previous_active))
        self.ctx.controller.deactivate_tool()

    def _cleanup(self, accept=False):
        if self._cleaned:
            return
        self._cleaned = True
        layer = self.preview_layer
        if layer is not None:
            panel = getattr(self.ctx.viewer.window(), "color_channels", None)
            if panel is not None and panel.layer is layer:
                if accept and panel.valid and panel.draft is not None and layer.preview_field is None:
                    panel.source_field.color_map = deepcopy(panel.draft)
                    panel.source_field.scalar_component = panel.field.scalar_component
                    panel.source_field.rgb_components = panel.field.rgb_components
                panel.dirty = False
                panel.activate(None)
            layer.preview_map_changed = None
            layer.preview_field = None
            layer._color_overrides.clear()
            if layer.id in self.ctx.scene.layers:
                self.ctx.scene.remove_layer(layer.id, delete_data=False)
            self.preview_layer = None
        renderer = self.ctx.viewer._renderer
        for actor in self._overlays:
            renderer.RemoveActor(actor)
        self._overlays.clear()
        for actor in self._spline_grid_actors+self._spline_control_actors:
            renderer.RemoveActor(actor)
        self._spline_grid_actors.clear()
        self._spline_control_actors.clear()
        if self._deviation_actor is not None:
            renderer.RemoveActor(self._deviation_actor)
            self._deviation_actor = None
            self._deviation_poly = None
        self._surface = self._deviation_field = None
        if hasattr(self, "source"):
            self.source.set_visible(self.source_visible)
        if getattr(self, "previous_active", None) in self.ctx.scene.layers:
            self.ctx.scene.active_layer_id = self.previous_active
        _refresh_scene(self.ctx)

    def deactivate(self):
        self._cancelled = True
        if self._worker is not None:
            self._worker.requestInterruption()
        self._cleanup()
        super().deactivate()

    def key_press_hook(self, event):
        if event.key and event.key.lower() == "escape":
            self.ctx.controller.deactivate_tool()
