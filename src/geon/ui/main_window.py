from .dataset_manager import Dock, DatasetManager
from .scene_manager import SceneManager
from .viewer import VTKViewer
from .toolbar import CommonToolsDock
from .menu_bar import MenuBar
from .context_ribbon import ContextRibbon
from .imports import FieldEditorDialog
from .preferences_dialog import PreferencesDialog
from .features_dialog import FeaturesDialog


from ..io.ply import ply_to_pcd
from ..algorithms.features import compute_pcd_features
from ..tools.controller import ToolController
from ..ui.layers import LAYER_UI
from ..rendering.pointcloud import PointCloudLayer
from ..data.pointcloud import FieldType, SemanticSegmentation, SemanticSchema
from geon.settings import Preferences
from geon.version import get_version
from geon.util.resources import resource_path


from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QMenu,
    QDialog,
    QLabel,
    QVBoxLayout,
    QProgressDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QEventLoop, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence, QAction, QIcon, QPixmap

from typing import cast
from geon._native import features as _native_features

class MainWindow(QMainWindow):
    def __init__(self, preferences: Preferences | None = None):
        super().__init__()
        self.preferences = preferences or Preferences.load()
        self.setWindowTitle("geon")
        
        QApplication.setApplicationName("geon")
        QApplication.setWindowIcon(QIcon(resource_path("geon_icon.png")))
        self.resize(1200,800)

        # settings
        self.setDockOptions(
                QMainWindow.DockOption.AllowTabbedDocks
            |   QMainWindow.DockOption.AllowNestedDocks
            |   QMainWindow.DockOption.GroupedDragging
        )        
        
        # widget initialization
        self.ribbon = ContextRibbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)

        self.viewer = VTKViewer(self)
        self.viewer.set_camera_sensitivity(self.preferences.camera_sensitivity)
        self.setCentralWidget(self.viewer)
        
        self.tool_controller = ToolController(context_ribbon=self.ribbon)
        self.tool_controller.install_tool_schortcuts(self)
        self.scene_manager = SceneManager(self.viewer, self.tool_controller, self) 
        self.scene_manager.preferences = self.preferences
        self.dataset_manager = DatasetManager(self)
        self.menu_bar = MenuBar(self)
        
        # menu bar
        view_menu = cast(QMenu, self.menu_bar.addMenu("&View"))
        view_menu.addAction(self.scene_manager.toggleViewAction())
        view_menu.addAction(self.dataset_manager.toggleViewAction())
        view_menu.addSeparator()
        act_toggle_edl = cast(QAction, view_menu.addAction("Toggle EDL"))
        act_toggle_edl.setCheckable(True)
        act_toggle_edl.toggled.connect(lambda checked: self.viewer.enable_edl() if checked else self.viewer.disable_edl())
        
        doc_menu = self.menu_bar.doc_menu
        doc_menu.addSeparator()
        import_field_menu = cast(QMenu, doc_menu.addMenu("Import field from ..."))
        act_import_npy = cast(QAction, import_field_menu.addAction(".NPY"))
        act_import_npy.triggered.connect(self._on_import_field_from_npy)
        act_edit_fields = cast(QAction, doc_menu.addAction("Edit fields"))
        act_edit_fields.triggered.connect(self._on_edit_fields)
        act_compute_features = cast(QAction, doc_menu.addAction("Compute geometric features"))
        act_compute_features.triggered.connect(self._on_compute_geometric_features)
        self.setMenuBar(self.menu_bar)

        ###########
        # signals #
        ###########
        
        self.scene_manager.broadcastDeleteScene\
            .connect(self.dataset_manager.save_scene_doc)

        self.dataset_manager.requestSetActiveDocInScene\
            .connect(self.scene_manager.on_document_loaded)
        self.dataset_manager.requestClearUndoStacks\
            .connect(self.tool_controller.clear_undo_stacks)
        
        self.menu_bar.setWorkdirRequested\
            .connect(self.dataset_manager.set_work_dir)
        self.menu_bar.importFromRequested\
            .connect(self.dataset_manager.import_doc_from_ply)
        self.menu_bar.saveDocRequested\
            .connect(lambda: self.dataset_manager.save_scene_doc(self.scene_manager._scene, ignore_state=True))
        self.menu_bar.undoRequested\
            .connect(lambda: self.tool_controller.command_manager.undo())
        self.menu_bar.redoRequested\
            .connect(lambda: self.tool_controller.command_manager.redo())
        self.menu_bar.editPreferencesRequested\
            .connect(self._on_edit_preferences)
        self.menu_bar.aboutRequested\
            .connect(self._on_about)
        
        self.tool_controller.tool_activated\
            .connect(lambda w: self.ribbon.set_group(self.tool_controller.active_tool_tooltip, w,'tool'))
        self.tool_controller.tool_activated\
            .connect(lambda _ :self.viewer.on_tool_activation(self.tool_controller.active_tool))
        self.tool_controller.tool_deactivated\
            .connect(lambda :self.viewer.on_tool_deactivation())
        self.tool_controller.tool_activated\
            .connect(lambda _w: self.scene_manager.log_tool_event(self.tool_controller.active_tool, "activated"))
        self.tool_controller.tool_deactivated\
            .connect(lambda : self.scene_manager.log_tool_event(self.tool_controller.last_tool, "deactivated"))
        self.tool_controller.scene_tree_request_change\
            .connect(self.scene_manager.populate_tree)    
        
            
        self.scene_manager.broadcastActivatedLayer\
            .connect(self._on_layer_activated)
        self.scene_manager.broadcastActivatedPcdField\
            .connect(self._on_layer_activated)

            
        self.tool_controller.layer_internal_sel_changed\
            .connect(self._on_layer_internal_sel_changed)
        # self.tool_controller.tool_activated\
        #     .connect(lambda _: self.viewer.tool_active_frame.show())
        # self.tool_controller.tool_deactivated\
        #     .connect(lambda: self.viewer.tool_active_frame.hide())


        # built-in shortcuts
        pass
        # escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        # escape_shortcut.activated.connect(self.tool_controller.deactivate_tool)
        

        self.tool_dock = CommonToolsDock("Tools", self, self.tool_controller)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.tool_dock)
        
        # initial float widget placement
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_manager)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dataset_manager)
        # self.tabifyDockWidget(self.scene_widget, self.dataset_widget)



        


              
        
        
    def _on_layer_activated(self, layer) -> None:
        hooks = LAYER_UI.resolve(layer)
        if hooks.ribbon_widget is None:
            self.ribbon.clear_group("layer")
            return
        title = getattr(layer, "browser_name", "Layer")
        widget = hooks.ribbon_widget(layer, self.ribbon, self.tool_controller)
        self.ribbon.set_group(title, widget, "layer")
        
    def _on_layer_internal_sel_changed(self, layer) -> None:
        hooks = LAYER_UI.resolve(layer)
        if hooks.ribbon_sel_widget is None:
            self.ribbon.clear_group('selection')
            return
        title = "Active selection"
        widget = hooks.ribbon_sel_widget(layer, self.ribbon, self.tool_controller)
        self.ribbon.set_group(title, widget, 'selection')

    def _get_active_pointcloud_layer(self) -> PointCloudLayer | None:
        scene = self.scene_manager._scene
        if scene is None:
            return None
        layer = scene.active_layer
        if not isinstance(layer, PointCloudLayer):
            return None
        return layer

    def _collect_semantic_schemas(self, layer: PointCloudLayer) -> dict[str, SemanticSchema]:
        schemas: dict[str, SemanticSchema] = {}
        dataset = self.dataset_manager._dataset
        if dataset is not None:
            for schema in dataset.unique_semantic_schemas:
                schemas[schema.name] = schema
            return schemas

        for field in layer.data.get_fields(field_type=FieldType.SEMANTIC):
            if isinstance(field, SemanticSegmentation):
                schemas[field.schema.name] = field.schema
        return schemas

    def _on_import_field_from_npy(self) -> None:
        layer = self._get_active_pointcloud_layer()
        if layer is None:
            return
        dlg = FieldEditorDialog.from_npy_picker(
            parent=self,
            semantic_schemas=self._collect_semantic_schemas(layer),
            color_maps={},
            target_point_cloud=layer.data,
        )
        if dlg is None:
            return
        dlg.exec()
        if dlg.point_cloud is None:
            return
        layer.update()
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_edit_preferences(self) -> None:
        dlg = PreferencesDialog(self.preferences, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.apply()
            self.preferences.save()
            self.scene_manager.preferences = self.preferences
            self.viewer.set_camera_sensitivity(self.preferences.camera_sensitivity)

    def _on_about(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("About geon")
        layout = QVBoxLayout(dlg)
        pix = QPixmap(resource_path("logo/geometric-red.png"))
        img_label = QLabel(dlg)
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_label)
        version_label = QLabel(f"Version: {get_version()}", dlg)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        dlg.setModal(True)
        dlg.exec()

    def _on_edit_fields(self) -> None:
        layer = self._get_active_pointcloud_layer()
        if layer is None:
            return
        dlg = FieldEditorDialog(
            ply_path=None,
            semantic_schemas=self._collect_semantic_schemas(layer),
            color_maps={},
            target_point_cloud=layer.data,
            edit_only=True,
            parent=self,
        )
        dlg.exec()
        if dlg.point_cloud is None:
            return
        layer.update()
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_compute_geometric_features(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            return
        active_layer = self._get_active_pointcloud_layer()
        dlg = FeaturesDialog(scene, active_layer, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        layer = dlg.selected_layer()
        if layer is None:
            return

        progress = _native_features.Progress()

        class _FeatureWorker(QThread):
            errored = pyqtSignal(str)

            def run(self) -> None:
                try:
                    compute_pcd_features(
                        radius=dlg.radius(),
                        data=layer.data,
                        field_name_normals=dlg.normals_field_name(),
                        field_name_eigenvals=dlg.eigenvals_field_name(),
                        compute_normals=dlg.compute_normals(),
                        compute_eigenvals=dlg.compute_eigenvals(),
                        progress=progress,
                    )
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog(
            "Computing geometric features...", "Cancel", 0, 0, self
        )
        progress_dialog.setWindowTitle("Compute geometric features")
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setMinimumDuration(0)

        def _on_cancel() -> None:
            progress.request_cancel()
            progress_dialog.setLabelText("Cancelling...")

        progress_dialog.canceled.connect(_on_cancel)

        timer = QTimer(self)
        timer.setInterval(100)

        def _on_tick() -> None:
            total = progress.total()
            done = progress.done()
            if total > 0:
                progress_dialog.setMaximum(int(total))
                progress_dialog.setValue(min(int(done), int(total)))
                progress_dialog.setLabelText(
                    f"Computing geometric features... {done}/{total}"
                )

        timer.timeout.connect(_on_tick)

        loop = QEventLoop(self)
        error_msg: dict[str, str | None] = {"value": None}

        def _on_finished() -> None:
            timer.stop()
            progress_dialog.close()
            loop.quit()

        def _on_error(msg: str) -> None:
            error_msg["value"] = msg
            _on_finished()

        worker = _FeatureWorker()
        worker.finished.connect(_on_finished)
        worker.errored.connect(_on_error)
        worker.start()
        timer.start()
        progress_dialog.show()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self, "Feature computation failed", error_msg["value"])
            return

        if progress.cancelled():
            return

        layer.update()
        self.scene_manager.populate_tree()
        self.viewer.rerender()
 
