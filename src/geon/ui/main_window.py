from .dataset_manager import Dock, DatasetManager
from .scene_manager import SceneManager
from .viewer import VTKViewer
from .toolbar import CommonToolsDock
from .menu_bar import MenuBar
from .layer_checks import require_active_layer
from .context_ribbon import ContextRibbon
from .imports import FieldEditorDialog
from .preferences_dialog import PreferencesDialog
from .features_dialog import FeaturesDialog
from .region_growing_dialog import RegionGrowingDialog
from .plane_ransac_dialog import PlaneRansacDialog
from .superpoint_segmentation_dialog import SuperpointSegmentationDialog
from .region_merge_dialog import RegionMergeDialog
from .corner_cleanup_dialog import CornerCleanupDialog
from .connected_components_dialog import ConnectedComponentsDialog
from .fit_obb_dialog import FitObbDialog
from .export_ply_dialog import PlyExportDialog


from ..io.ply import pcd_to_ply, ply_to_pcd
from ..algorithms.features import compute_pcd_features
from ..algorithms.region_growing import (
    estimate_parameters as estimate_region_growing_parameters,
    segment_planar_regions,
)
from ..algorithms.plane_ransac import segment_planes as segment_ransac_planes
from ..algorithms.superpoints import segment_superpoints
from ..algorithms.region_merge import merge_planar_regions
from ..algorithms.corner_cleanup import cleanup_corner_regions
from ..algorithms.connected_components import segment_connected_components
from ..algorithms.fit_obb import fit_obb
from ..tools.controller import ToolController
from ..tools.boundingbox import AddBoundingBoxesCmd
from ..ui.layers import LAYER_UI
from ..rendering.pointcloud import PointCloudLayer
from ..rendering.cellcomplex import CellComplexLayer
from ..rendering.boundingbox import BoundingBoxLayer
from ..data.pointcloud import (
    FieldType,
    InstanceSegmentation,
    SemanticSegmentation,
    SemanticSchema,
)
from ..data.cellcomplex import CellComplexData
from ..data.camera import CameraData
from ..data.boundingbox import BoundingBoxData
from ..data.document import Document
from ..io.dataset import RefModState
from geon.settings import Preferences
from geon.version import get_version
from geon.util.resources import resource_path


from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QScrollArea,
    QProgressBar,
    QPushButton,
    QProgressDialog,
    QMessageBox,
    QSizePolicy,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QEventLoop, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence, QAction, QIcon, QPixmap

from typing import cast
from geon._native import features as _native_features
from geon._native import region_growing as _native_region_growing
from geon._native import plane_ransac as _native_plane_ransac
from geon._native import superpoints as _native_superpoints
from geon._native import region_merge as _native_region_merge
from geon._native import corner_cleanup as _native_corner_cleanup
import numpy as np
import time
import weakref


class _RenameSceneDialog(QDialog):
    def __init__(self, current_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Scene")

        layout = QVBoxLayout(self)
        label = QLabel("Scene name", self)
        layout.addWidget(label)

        self.name_edit = QLineEdit(self)
        self.name_edit.setText(current_name)
        self.name_edit.selectAll()
        layout.addWidget(self.name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scene_name(self) -> str:
        return self.name_edit.text().strip()


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
        
        self._populate_point_cloud_menu()
        self.setMenuBar(self.menu_bar)

        ###########
        # signals #
        ###########
        
        self.scene_manager.broadcastDeleteScene\
            .connect(self.dataset_manager.save_scene_doc)

        self.dataset_manager.requestSetActiveDocInScene\
            .connect(self.scene_manager.on_document_loaded)
        self.dataset_manager.requestSetActiveDocInScene\
            .connect(lambda _doc: (self._apply_cell_complex_preferences(), self._apply_bounding_box_preferences(), self.viewer.rerender()))
        self.dataset_manager.requestClearUndoStacks\
            .connect(self.tool_controller.clear_undo_stacks)
        
        self.menu_bar.setWorkdirRequested\
            .connect(self.dataset_manager.set_work_dir)
        self.menu_bar.importFromRequested\
            .connect(self.dataset_manager.import_doc_from_ply)
        self.menu_bar.importCellComplexTxtRequested\
            .connect(self._on_import_cell_complex_from_txt)
        self.menu_bar.exportPointCloudPlyRequested\
            .connect(self._on_export_active_point_cloud_to_ply)
        self.menu_bar.saveDocRequested\
            .connect(lambda: self.dataset_manager.save_scene_doc(self.scene_manager._scene, ignore_state=True))
        self.menu_bar.renderToFileRequested\
            .connect(self._on_render_to_file)
        self.menu_bar.renameSceneRequested\
            .connect(self._on_rename_scene)
        self.menu_bar.createCameraSnapshotRequested\
            .connect(self._on_create_camera_snapshot)
        self.menu_bar.importCameraSnapshotJsonRequested\
            .connect(self._on_import_camera_snapshot_from_json)
        self.menu_bar.createEmptyBoundingBoxLayerRequested\
            .connect(self._on_create_empty_bounding_box_layer)
        self.menu_bar.fitObbFromSelectionRequested\
            .connect(self._on_fit_obb_from_selection)
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
        self.tool_controller.layer_data_modified.connect(self._on_layer_data_modified)
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
        else:
            title = getattr(layer, "browser_name", "Layer")
            widget = hooks.ribbon_widget(layer, self.ribbon, self.tool_controller)
            self.ribbon.set_group(title, widget, "layer")
        self._on_layer_internal_sel_changed(layer)
        
    def _on_layer_internal_sel_changed(self, layer) -> None:
        hooks = LAYER_UI.resolve(layer)
        if hooks.ribbon_sel_widget is None:
            self.ribbon.clear_group('selection')
            return
        title = "Active selection"
        widget = hooks.ribbon_sel_widget(layer, self.ribbon, self.tool_controller)
        self.ribbon.set_group(title, widget, 'selection')

    def _on_layer_data_modified(self, _layer) -> None:
        scene = self.scene_manager._scene
        dataset = self.dataset_manager._dataset
        if scene is None or dataset is None:
            return
        for ref in dataset.doc_refs:
            if ref.name == scene.doc.name:
                ref.modState = RefModState.MODIFIED
                break

    def _populate_point_cloud_menu(self) -> None:
        point_cloud_menu = self.menu_bar.point_cloud_menu
        import_field_menu = cast(QMenu, point_cloud_menu.addMenu("Import field from ..."))
        act_import_npy = cast(QAction, import_field_menu.addAction(".NPY"))
        act_import_npy.triggered.connect(self._on_import_field_from_npy)
        act_edit_fields = cast(QAction, point_cloud_menu.addAction("Edit fields"))
        act_edit_fields.triggered.connect(self._on_edit_fields)
        act_compute_features = cast(QAction, point_cloud_menu.addAction("Compute geometric features"))
        act_compute_features.triggered.connect(self._on_compute_geometric_features)
        seg_menu = cast(QMenu, point_cloud_menu.addMenu("Segmentation"))
        act_planar_region_growing = cast(QAction, seg_menu.addAction("Planar region growing"))
        act_planar_region_growing.triggered.connect(self._on_planar_region_growing)
        act_plane_ransac = cast(QAction, seg_menu.addAction("Plane RANSAC"))
        act_plane_ransac.triggered.connect(self._on_plane_ransac)
        act_superpoints = cast(QAction, seg_menu.addAction("Superpoint segmentation"))
        act_superpoints.triggered.connect(self._on_superpoint_segmentation)
        act_region_merge = cast(QAction, seg_menu.addAction("Merge planar regions"))
        act_region_merge.triggered.connect(self._on_region_merge)
        act_corner_cleanup = cast(QAction, seg_menu.addAction("Clean up corner regions"))
        act_corner_cleanup.triggered.connect(self._on_corner_cleanup)
        act_connected_components = cast(
            QAction, seg_menu.addAction("Connected components instance segmentation")
        )
        act_connected_components.triggered.connect(self._on_connected_components)

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
        layer = require_active_layer(self, self.scene_manager._scene, PointCloudLayer)
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

    def _on_rename_scene(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            QMessageBox.warning(self, "Rename Scene", "No scene is currently loaded.")
            return

        dlg = _RenameSceneDialog(scene.doc.name, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_name = dlg.scene_name()
        if not new_name:
            QMessageBox.warning(self, "Rename Scene", "Scene name cannot be empty.")
            return

        dataset = self.dataset_manager._dataset
        if dataset is not None:
            current_name = scene.doc.name
            if new_name != current_name and new_name in dataset.doc_ref_names:
                QMessageBox.warning(
                    self,
                    "Rename Scene",
                    f"A scene named '{new_name}' already exists in the dataset.",
                )
                return
            try:
                dataset.rename_document(scene.doc, new_name)
            except Exception as exc:
                QMessageBox.critical(self, "Rename Scene", str(exc))
                return
            self.dataset_manager.populate_tree()
        else:
            scene.doc.name = new_name
            scene.doc.meta["name"] = new_name

        self.scene_manager.update_tree_visibility()
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_edit_preferences(self) -> None:
        dlg = PreferencesDialog(self.preferences, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.apply()
            self.preferences.save()
            self.scene_manager.preferences = self.preferences
            self.viewer.set_camera_sensitivity(self.preferences.camera_sensitivity)
            self._apply_cell_complex_preferences()
            self._apply_bounding_box_preferences()
            self.viewer.rerender()

    def _apply_cell_complex_preferences(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            return
        default_color = tuple(int(c) for c in self.preferences.cell_complex_default_color)
        selection_color = tuple(int(c) for c in self.preferences.selection_color)
        for layer in scene.layers.values():
            if isinstance(layer, CellComplexLayer):
                layer.set_visual_settings(
                    size_mode=self.preferences.cell_complex_size_mode,
                    screen_size_px=self.preferences.cell_complex_screen_size_px,
                    world_size=self.preferences.cell_complex_world_size,
                    edge_width=self.preferences.cell_complex_edge_width,
                    reference_label_text_size_px=(
                        self.preferences.cell_complex_reference_label_text_size_px
                    ),
                    default_color=default_color,
                    selection_color=selection_color,
                )

    def _apply_bounding_box_preferences(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            return
        default_color = tuple(int(c) for c in self.preferences.cell_complex_default_color)
        selection_color = tuple(int(c) for c in self.preferences.selection_color)
        for layer in scene.layers.values():
            if isinstance(layer, BoundingBoxLayer):
                layer.set_visual_settings(
                    default_color=default_color,
                    selection_color=selection_color,
                )

    def _mark_active_doc_modified(self) -> None:
        dataset = self.dataset_manager._dataset
        scene = self.scene_manager._scene
        if dataset is None or scene is None:
            return
        for ref in dataset.doc_refs:
            if ref.name == scene.doc.name:
                ref.modState = RefModState.MODIFIED
                break

    def _on_create_camera_snapshot(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            QMessageBox.information(
                self,
                "Camera Snapshot",
                "Load a scene before creating a camera snapshot.",
            )
            return
        camera_snapshot = CameraData.from_camera(self.viewer._renderer.GetActiveCamera())
        scene.doc.add_data(camera_snapshot)
        layer = scene.add_data(camera_snapshot)
        scene.active_layer_id = layer.id
        self._mark_active_doc_modified()
        self.scene_manager.populate_tree()
        self.dataset_manager.populate_tree()
        self.scene_manager.broadcastActivatedLayer.emit(layer)
        self.viewer.rerender()

    def _on_import_camera_snapshot_from_json(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            QMessageBox.information(
                self,
                "Import Camera Snapshot",
                "Load a scene before importing a camera snapshot.",
            )
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import Camera Snapshot JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            camera_snapshot = CameraData.load_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Camera Snapshot", str(exc))
            return
        scene.doc.add_data(camera_snapshot)
        layer = scene.add_data(camera_snapshot)
        scene.active_layer_id = layer.id
        self._mark_active_doc_modified()
        self.scene_manager.populate_tree()
        self.dataset_manager.populate_tree()
        self.scene_manager.broadcastActivatedLayer.emit(layer)
        self.viewer.rerender()

    def _on_export_active_point_cloud_to_ply(self) -> None:
        scene = self.scene_manager._scene
        if scene is None or not isinstance(scene.active_layer, PointCloudLayer):
            QMessageBox.information(
                self,
                "Export PLY",
                "Select a point cloud layer before exporting to PLY.",
            )
            return

        layer = scene.active_layer
        dlg = PlyExportDialog(layer.data, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        if not opts.path:
            QMessageBox.warning(self, "Export PLY", "Choose an output file.")
            return
        try:
            pcd_to_ply(
                layer.data,
                opts.path,
                field_names=opts.field_names,
                field_dtypes=opts.field_dtypes,
                coord_dtype=opts.coord_dtype,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export PLY", str(exc))
            return
        QMessageBox.information(self, "Export PLY", f"Exported point cloud to:\n{opts.path}")

    def _on_import_cell_complex_from_txt(self) -> None:
        if self.dataset_manager._dataset is None:
            success = self.dataset_manager.set_work_dir()
            if self.dataset_manager._dataset is None or not success:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CellComplex TXT File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        try:
            cell_complex = CellComplexData.from_txt(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Import CellComplex", str(exc))
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Import CellComplex")
        msg.setText("Where should the CellComplex be imported?")
        new_doc_btn = msg.addButton("New document", QMessageBox.ButtonRole.AcceptRole)
        active_doc_btn = msg.addButton("Active document", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        active_doc_btn.setEnabled(self.scene_manager._scene is not None)
        msg.setDefaultButton(new_doc_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is None or clicked is cancel_btn:
            return

        if clicked is active_doc_btn and self.scene_manager._scene is not None:
            scene = self.scene_manager._scene
            scene.doc.add_data(cell_complex)
            layer = scene.add_data(cell_complex)
            scene.active_layer_id = layer.id
            self._apply_cell_complex_preferences()
            self._mark_active_doc_modified()
            self.scene_manager.populate_tree()
            self.dataset_manager.populate_tree()
            self.scene_manager.broadcastActivatedLayer.emit(layer)
            self.viewer.rerender()
            return

        name_cand = file_path.replace("\\", "/").split("/")[-1]
        name = name_cand.rsplit(".", 1)[0]
        dataset = self.dataset_manager._dataset
        if dataset is None:
            return
        base_name = name
        suffix = 0
        while name in dataset.doc_ref_names:
            name = f"{base_name}_{suffix:03}"
            suffix += 1
        doc = Document(name)
        doc.add_data(cell_complex)
        doc_ref = dataset.add_document(doc)
        self.dataset_manager.populate_tree()
        self.dataset_manager.set_active_doc(doc_ref)

    def _on_create_empty_bounding_box_layer(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            QMessageBox.information(
                self,
                "Create Bounding Box Layer",
                "Load a scene before creating a bounding box layer.",
            )
            return
        data = BoundingBoxData([])
        scene.doc.add_data(data)
        layer = scene.add_data(data)
        scene.active_layer_id = layer.id
        self._apply_bounding_box_preferences()
        self._mark_active_doc_modified()
        self.scene_manager.populate_tree()
        self.dataset_manager.populate_tree()
        self.scene_manager.broadcastActivatedLayer.emit(layer)
        self.viewer.rerender()

    def _on_fit_obb_from_selection(self) -> None:
        scene = self.scene_manager._scene
        if scene is None:
            QMessageBox.information(
                self,
                "Fit oriented bounding box",
                "Load a scene before fitting a bounding box.",
            )
            return
        source_layer = self._get_active_pointcloud_layer()
        if source_layer is None:
            QMessageBox.information(
                self,
                "Fit oriented bounding box",
                "Activate a point-cloud layer first.",
            )
            return
        selection = source_layer.active_selection
        if not any(isinstance(layer, BoundingBoxLayer) for layer in scene.layers.values()):
            QMessageBox.information(
                self,
                "Fit oriented bounding box",
                "Create a bounding-box layer before fitting a box.",
            )
            return

        # Snapshot the current selection, falling back to the complete cloud.
        if selection is None or selection.size == 0:
            selected_indices = np.arange(
                source_layer.data.points.shape[0], dtype=np.int64
            )
        else:
            selected_indices = np.asarray(selection, dtype=np.int64).reshape(-1).copy()
        if (
            selected_indices.size == 0
            or int(selected_indices.min()) < 0
            or int(selected_indices.max()) >= source_layer.data.points.shape[0]
            or np.unique(selected_indices).size != selected_indices.size
        ):
            QMessageBox.warning(
                self,
                "Fit oriented bounding box",
                "The point cloud is empty or its active selection is invalid.",
            )
            return

        dialog = FitObbDialog(scene, source_layer, selected_indices.size, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target_layer = dialog.target_layer()
        if target_layer is None:
            return

        point_groups: list[tuple[int | None, np.ndarray]] = [(None, selected_indices)]
        separation_field_name = dialog.separation_field_name()
        if separation_field_name is not None:
            fields = source_layer.data.get_fields(names=separation_field_name)
            if (
                not fields
                or fields[0].field_type not in (FieldType.INSTANCE, FieldType.SEMANTIC)
            ):
                QMessageBox.warning(
                    self,
                    "Fit oriented bounding box",
                    "The selected separation field is no longer available.",
                )
                return
            field_values = np.asarray(fields[0].data).reshape(-1)
            if field_values.shape[0] != source_layer.data.points.shape[0]:
                QMessageBox.warning(
                    self,
                    "Fit oriented bounding box",
                    "The separation field length does not match the point cloud.",
                )
                return
            selected_values = field_values[selected_indices]
            point_groups = [
                (int(value), selected_indices[selected_values == value])
                for value in np.unique(selected_values)
            ]

        boxes = []
        try:
            for field_value, group_indices in point_groups:
                box = fit_obb(
                    source_layer.data.points[group_indices],
                    method=dialog.method(),
                    trim=dialog.trim(),
                )
                if separation_field_name is not None:
                    box.attributes.update({
                        "separation_field": separation_field_name,
                        "separation_value": field_value,
                    })
                boxes.append(box)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Fit oriented bounding box", str(exc))
            return

        self.tool_controller.do_global(
            AddBoundingBoxesCmd(
                "Fit oriented bounding boxes",
                weakref.ref(target_layer),
                boxes,
            )
        )
        target_layer.active_selection = {box.id for box in boxes}
        scene.active_layer_id = target_layer.id
        self._mark_active_doc_modified()
        self.scene_manager.populate_tree()
        self.dataset_manager.populate_tree()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.statusBar().showMessage(
            f"Added {len(boxes):,} fitted bounding box(es).",
            8000,
        )
        self.viewer.rerender()

    def _on_render_to_file(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Render Viewport to PNG",
            "viewport.png",
            "PNG Images (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self.viewer.save_viewport_png(path, transparent=True)
        except Exception as exc:
            QMessageBox.critical(self, "Render to File", str(exc))

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
        layer = require_active_layer(self, self.scene_manager._scene, PointCloudLayer)
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

    @staticmethod
    def _unique_field_name(existing: list[str], base: str) -> str:
        if base not in existing:
            return base
        suffix = 1
        while True:
            candidate = f"{base}_{suffix:03d}"
            if candidate not in existing:
                return candidate
            suffix += 1

    def _on_planar_region_growing(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dlg = RegionGrowingDialog(
            scene,
            active_layer,
            settings=self.preferences.get_region_growing_settings(),
            parent=self,
        )

        def _run_estimate() -> None:
            layer = dlg.selected_layer()
            if layer is None:
                return
            estimate_data = layer.data
            if dlg.on_selection_only():
                selection = layer.active_selection
                if selection is None or selection.size == 0:
                    QMessageBox.warning(
                        self,
                        "Planar region growing",
                        "The selected point cloud no longer has an active selection.",
                    )
                    return
                estimate_data, _ = layer.data.extract_subcloud(selection.copy())
            estimate_kwargs = dlg.estimate_kwargs()

            class _EstimateWorker(QThread):
                estimated = pyqtSignal(float, int, float)
                errored = pyqtSignal(str)

                def run(self) -> None:
                    try:
                        estimated = estimate_region_growing_parameters(
                            estimate_data,
                            **estimate_kwargs,
                        )
                        self.estimated.emit(
                            float(estimated["epsilon"]),
                            int(estimated["tau"]),
                            float(estimated["alpha_deg"]),
                        )
                    except Exception as exc:  # pragma: no cover - GUI path
                        self.errored.emit(str(exc))

            progress_dialog = QProgressDialog("Estimating parameters...", "", 0, 0, self)
            progress_dialog.setWindowTitle("Planar region growing")
            progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress_dialog.setCancelButton(None)
            progress_dialog.setMinimumDuration(0)

            loop = QEventLoop(self)
            error_msg: dict[str, str | None] = {"value": None}

            def _on_estimated(eps: float, tau: int, alpha_deg: float) -> None:
                dlg.set_estimated_core(eps, tau, alpha_deg)
                progress_dialog.close()
                loop.quit()

            def _on_error(msg: str) -> None:
                error_msg["value"] = msg
                progress_dialog.close()
                loop.quit()

            worker = _EstimateWorker()
            worker.estimated.connect(_on_estimated)
            worker.errored.connect(_on_error)
            worker.finished.connect(lambda: None)
            worker.start()
            progress_dialog.show()
            loop.exec()
            worker.wait()

            if error_msg["value"] is not None:
                QMessageBox.critical(self, "Parameter estimation failed", error_msg["value"])

        dlg.estimate_button.clicked.connect(_run_estimate)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dlg.selected_layer()
        if layer is None:
            return
        layer_id = layer.id
        normal_mode = dlg.normal_mode()
        if normal_mode is None:
            return
        selection_only = dlg.on_selection_only()
        output_mode = dlg.output_mode()
        output_field_name: str
        target_instance_field: InstanceSegmentation | None = None
        if output_mode == "write_existing":
            existing_field_name = dlg.existing_field_name()
            if existing_field_name is None:
                QMessageBox.warning(
                    self,
                    "Planar region growing",
                    "No output instance field was selected.",
                )
                return
            existing_fields = layer.data.get_fields(
                names=existing_field_name,
                field_type=FieldType.INSTANCE,
            )
            if not existing_fields or not isinstance(existing_fields[0], InstanceSegmentation):
                QMessageBox.warning(
                    self,
                    "Planar region growing",
                    "The selected output instance field was not found.",
                )
                return
            target_instance_field = existing_fields[0]
            output_field_name = target_instance_field.name
        else:
            base_name = dlg.output_field_base() or "planar_regions"
            output_field_name = self._unique_field_name(layer.data.field_names, base_name)

        self.preferences.set_region_growing_settings(dlg.settings())
        self.preferences.save()

        segmentation_data = layer.data
        inverse_mapping: np.ndarray | None = None
        normal_field_name = dlg.normal_field_name() if normal_mode == "use_provided" else None
        if selection_only:
            selection = layer.active_selection
            if selection is None or selection.size == 0:
                QMessageBox.warning(
                    self,
                    "Planar region growing",
                    "The selected point cloud no longer has an active selection.",
                )
                return
            requested_fields = [normal_field_name] if normal_field_name is not None else []
            try:
                segmentation_data, inverse_mapping = layer.data.extract_subcloud(
                    selection.copy(),
                    field_names=requested_fields,
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Planar region growing", str(exc))
                return

        normals: np.ndarray | None = None
        if normal_mode == "use_provided":
            if normal_field_name is None:
                QMessageBox.warning(self, "Planar region growing", "No normals field selected.")
                return
            normal_fields = segmentation_data.get_fields(
                names=normal_field_name,
                field_type=FieldType.NORMAL,
            )
            if not normal_fields:
                QMessageBox.warning(self, "Planar region growing", "Selected normals field was not found.")
                return
            normals = np.asarray(normal_fields[0].data, dtype=np.float32)
            if normals.ndim != 2 or normals.shape[1] != 3:
                QMessageBox.warning(
                    self,
                    "Planar region growing",
                    f"Normals field '{normal_field_name}' must have shape (N,3).",
                )
                return

        progress = _native_region_growing.Progress()
        region_params = dlg.params()
        chunking_params = dlg.chunking()
        merge_params = dlg.merge()
        result_labels: dict[str, np.ndarray | None] = {"value": None}

        class _RegionGrowingWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, stats = segment_planar_regions(
                        segmentation_data,
                        normals=normals,
                        normal_mode=normal_mode,
                        params=region_params,
                        chunking=chunking_params,
                        merge=merge_params,
                        progress=progress,
                    )
                    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
                    if labels.shape[0] != segmentation_data.points.shape[0]:
                        raise RuntimeError("Native output label length does not match point count.")
                    result_labels["value"] = labels
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Planar region growing")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.resize(760, 560)
        progress_layout = QVBoxLayout(progress_dialog)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(8)

        overall_label = QLabel("Running planar region growing...", progress_dialog)
        overall_label.setWordWrap(False)
        overall_label.setFixedHeight(overall_label.fontMetrics().height() + 6)
        overall_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        progress_layout.addWidget(overall_label)
        overall_bar = QProgressBar(progress_dialog)
        overall_bar.setRange(0, 0)
        overall_bar.setValue(0)
        overall_bar.setFixedHeight(overall_bar.sizeHint().height())
        overall_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        progress_layout.addWidget(overall_bar)

        def _on_cancel() -> None:
            progress.request_cancel()
            overall_label.setText("Cancelling...")

        chunk_scroll = QScrollArea(progress_dialog)
        chunk_scroll.setWidgetResizable(True)
        chunk_scroll.setMinimumWidth(560)
        chunk_scroll.setMinimumHeight(420)
        chunk_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chunk_container = QWidget(chunk_scroll)
        chunk_rows_layout = QVBoxLayout(chunk_container)
        chunk_rows_layout.setContentsMargins(4, 4, 4, 4)
        chunk_rows_layout.setSpacing(6)
        chunk_rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        chunk_scroll.setWidget(chunk_container)
        progress_layout.addWidget(chunk_scroll, 1)

        cancel_btn = QPushButton("Cancel", progress_dialog)
        cancel_btn.clicked.connect(_on_cancel)
        progress_layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)
        chunk_rows: dict[int, tuple[QLabel, QProgressBar, QLabel]] = {}

        def _ensure_chunk_row(chunk_idx: int) -> tuple[QLabel, QProgressBar, QLabel]:
            existing = chunk_rows.get(chunk_idx)
            if existing is not None:
                return existing
            row = QWidget(chunk_container)
            row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            label = QLabel(f"Chunk {chunk_idx}", row)
            label.setFixedWidth(80)
            bar = QProgressBar(row)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedWidth(260)
            bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            info = QLabel("-", row)
            info.setFixedWidth(240)
            info.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row_layout.addWidget(label)
            row_layout.addWidget(bar)
            row_layout.addWidget(info)
            row_layout.addStretch(1)
            chunk_rows_layout.addWidget(row)
            chunk_rows[chunk_idx] = (label, bar, info)
            return chunk_rows[chunk_idx]

        timer = QTimer(self)
        timer.setInterval(100)
        t_start = time.perf_counter()
        tick_counter = 0

        def _on_tick() -> None:
            nonlocal tick_counter
            tick_counter += 1
            try:
                total = progress.total()
                done = progress.done()
                elapsed = time.perf_counter() - t_start
                if total > 0:
                    overall_bar.setRange(0, int(total))
                    overall_bar.setValue(min(int(done), int(total)))
                    overall_label.setText(
                        f"Running planar region growing... {done}/{total} | {elapsed:.1f}s"
                    )
                else:
                    overall_bar.setRange(0, 0)
                    overall_label.setText(
                        f"Running planar region growing... {elapsed:.1f}s"
                    )

                # Poll chunk telemetry at lower frequency to reduce overhead.
                if (tick_counter % 5) != 0:
                    return
                if not hasattr(progress, "chunk_statuses"):
                    return
                statuses = progress.chunk_statuses()
                for s in statuses:
                    chunk_idx = int(s.get("chunk", -1))
                    if chunk_idx < 0:
                        continue
                    _, bar, info = _ensure_chunk_row(chunk_idx)
                    fail_rate = float(s.get("fail_rate", 0.0))
                    fail_threshold = max(1e-6, float(s.get("fail_threshold", 1.0)))
                    rel = min(1.5, fail_rate / fail_threshold)
                    bar.setValue(int(min(100.0, rel * 100.0)))
                    attempts = int(s.get("attempts", 0))
                    regions = int(s.get("regions", 0))
                    remaining = int(s.get("remaining", 0))
                    phase = int(s.get("phase", 0))
                    if phase <= 0:
                        state = "segmenting"
                    elif phase == 1:
                        state = "finalizing"
                    else:
                        state = "done"
                    info.setText(
                        f"{state} | a={attempts} r={regions} rem={remaining} "
                        f"f={fail_rate:.2f}/{fail_threshold:.2f}"
                    )
            except Exception as exc:  # pragma: no cover - GUI path
                # Keep UI alive and expose polling failures for debugging.
                print(f"[region_growing/ui] progress polling failed: {exc}")

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

        def _on_completed() -> None:
            _on_finished()

        worker = _RegionGrowingWorker()
        worker.errored.connect(_on_error)
        worker.completed.connect(_on_completed)
        worker.finished.connect(lambda: None)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self, "Planar region growing failed", error_msg["value"])
            return

        if progress.cancelled():
            return

        labels = result_labels["value"]
        if labels is None:
            QMessageBox.critical(
                self,
                "Planar region growing failed",
                "Region growing completed without producing labels.",
            )
            return

        try:
            if target_instance_field is not None:
                target_instance_field.merge_with_segmentation(labels, inverse_mapping)
                output_action = "Updated"
            else:
                if inverse_mapping is None:
                    output_labels = labels
                else:
                    output_labels = np.full(layer.data.points.shape[0], -1, dtype=np.int32)
                    output_labels[inverse_mapping] = labels
                layer.data.add_field(
                    name=output_field_name,
                    data=output_labels[:, None],
                    field_type=FieldType.INSTANCE,
                )
                output_action = "Added"
        except Exception as exc:  # pragma: no cover - defensive GUI path
            QMessageBox.critical(
                self,
                "Planar region growing failed",
                f"Could not write segmentation output: {exc}",
            )
            return

        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            target_layer_obj = scene_after.layers.get(layer_id)
            if isinstance(target_layer_obj, PointCloudLayer):
                target_layer = target_layer_obj
                scene_after.active_layer_id = target_layer.id

        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        print(
            f"[region_growing] {output_action} field '{output_field_name}' "
            f"(type=INSTANCE, points={target_layer.data.points.shape[0]}) on layer '{target_layer.browser_name}'"
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_plane_ransac(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dlg = PlaneRansacDialog(
            scene,
            active_layer,
            settings=self.preferences.get_plane_ransac_settings(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dlg.selected_layer()
        if layer is None:
            return
        self.preferences.set_plane_ransac_settings(dlg.settings())
        self.preferences.save()
        layer_id = layer.id
        normal_mode = dlg.normal_mode()
        if normal_mode is None:
            return

        normals: np.ndarray | None = None
        if normal_mode == "use_provided":
            normal_field_name = dlg.normal_field_name()
            if normal_field_name is None:
                QMessageBox.warning(self, "Plane RANSAC", "No normals field selected.")
                return
            normal_fields = layer.data.get_fields(names=normal_field_name, field_type=FieldType.NORMAL)
            if not normal_fields:
                QMessageBox.warning(self, "Plane RANSAC", "Selected normals field was not found.")
                return
            normals = np.asarray(normal_fields[0].data, dtype=np.float32)
            if normals.ndim != 2 or normals.shape[1] != 3:
                QMessageBox.warning(
                    self,
                    "Plane RANSAC",
                    f"Normals field '{normal_field_name}' must have shape (N,3).",
                )
                return

        progress = _native_plane_ransac.Progress()
        output_field_name = self._unique_field_name(
            layer.data.field_names,
            dlg.output_field_base() or "ransac_planes",
        )

        class _PlaneRansacWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, _stats = segment_ransac_planes(
                        layer.data,
                        normals=normals,
                        normal_mode=normal_mode,
                        params=dlg.params(),
                        progress=progress,
                    )
                    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
                    if labels.shape[0] != layer.data.points.shape[0]:
                        raise RuntimeError("Native output label length does not match point count.")
                    layer.data.add_field(
                        name=output_field_name,
                        data=labels[:, None],
                        field_type=FieldType.INSTANCE,
                    )
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog("Running plane RANSAC...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Plane RANSAC")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)

        cancel_requested = {"value": False}

        def _on_cancel() -> None:
            cancel_requested["value"] = True
            progress.request_cancel()
            progress_dialog.setLabelText("Cancelling...")

        progress_dialog.canceled.connect(_on_cancel)

        timer = QTimer(self)
        timer.setInterval(100)
        t_start = time.perf_counter()

        def _on_tick() -> None:
            try:
                total = progress.total()
                done = progress.done()
                elapsed = time.perf_counter() - t_start
                if total > 0:
                    progress_dialog.setMaximum(int(total))
                    progress_dialog.setValue(min(int(done), int(total)))
                stage = progress.stage()
                progress_dialog.setLabelText(
                    f"{stage} | planes={progress.planes_found()} "
                    f"| remaining={progress.active_points_remaining()} "
                    f"| best={progress.current_best_support()} "
                    f"| {elapsed:.1f}s"
                )
            except Exception as exc:  # pragma: no cover - GUI path
                print(f"[plane_ransac/ui] progress polling failed: {exc}")

        timer.timeout.connect(_on_tick)

        loop = QEventLoop(self)
        error_msg: dict[str, str | None] = {"value": None}

        def _on_finished() -> None:
            timer.stop()
            try:
                progress_dialog.canceled.disconnect(_on_cancel)
            except TypeError:
                pass
            progress_dialog.close()
            loop.quit()

        def _on_error(msg: str) -> None:
            error_msg["value"] = msg
            _on_finished()

        worker = _PlaneRansacWorker()
        worker.errored.connect(_on_error)
        worker.completed.connect(_on_finished)
        worker.finished.connect(lambda: None)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self, "Plane RANSAC failed", error_msg["value"])
            return

        if cancel_requested["value"]:
            return

        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            target_layer_obj = scene_after.layers.get(layer_id)
            if isinstance(target_layer_obj, PointCloudLayer):
                target_layer = target_layer_obj
                scene_after.active_layer_id = target_layer.id

        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        print(
            f"[plane_ransac] Added field '{output_field_name}' "
            f"(type=INSTANCE, points={target_layer.data.points.shape[0]}) on layer '{target_layer.browser_name}'"
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_superpoint_segmentation(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dlg = SuperpointSegmentationDialog(
            scene,
            active_layer,
            settings=self.preferences.get_superpoints_settings(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dlg.selected_layer()
        if layer is None:
            return
        self.preferences.set_superpoints_settings(dlg.settings())
        self.preferences.save()
        layer_id = layer.id
        output_field_name = self._unique_field_name(
            layer.data.field_names,
            dlg.output_field_base() or "superpoints",
        )
        feature_field_names = dlg.selected_feature_field_names()
        progress = _native_superpoints.Progress()

        class _SuperpointsWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, _stats = segment_superpoints(
                        layer.data,
                        feature_field_names=feature_field_names,
                        progress=progress,
                        **dlg.params(),
                    )
                    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
                    if labels.shape[0] != layer.data.points.shape[0]:
                        raise RuntimeError("Native output label length does not match point count.")
                    layer.data.add_field(
                        name=output_field_name,
                        data=labels[:, None],
                        field_type=FieldType.INSTANCE,
                    )
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog("Preparing superpoint segmentation...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Superpoint segmentation")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)

        cancel_requested = {"value": False}

        def _on_cancel() -> None:
            cancel_requested["value"] = True
            progress.request_cancel()
            progress_dialog.setLabelText("Cancelling...")

        progress_dialog.canceled.connect(_on_cancel)

        timer = QTimer(self)
        timer.setInterval(100)
        t_start = time.perf_counter()

        def _on_tick() -> None:
            try:
                total = progress.total()
                done = progress.done()
                elapsed = time.perf_counter() - t_start
                if total > 0:
                    progress_dialog.setMaximum(int(total))
                    progress_dialog.setValue(min(int(done), int(total)))
                else:
                    progress_dialog.setMaximum(0)
                progress_dialog.setLabelText(
                    f"{progress.stage()} | {elapsed:.1f}s"
                )
            except Exception as exc:  # pragma: no cover - GUI path
                print(f"[superpoints/ui] progress polling failed: {exc}")

        timer.timeout.connect(_on_tick)

        loop = QEventLoop(self)
        error_msg: dict[str, str | None] = {"value": None}

        def _on_finished() -> None:
            timer.stop()
            try:
                progress_dialog.canceled.disconnect(_on_cancel)
            except TypeError:
                pass
            progress_dialog.close()
            loop.quit()

        def _on_error(msg: str) -> None:
            error_msg["value"] = msg
            _on_finished()

        worker = _SuperpointsWorker()
        worker.errored.connect(_on_error)
        worker.completed.connect(_on_finished)
        worker.finished.connect(lambda: None)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self, "Superpoint segmentation failed", error_msg["value"])
            return
        if cancel_requested["value"]:
            return

        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            target_layer_obj = scene_after.layers.get(layer_id)
            if isinstance(target_layer_obj, PointCloudLayer):
                target_layer = target_layer_obj
                scene_after.active_layer_id = target_layer.id

        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        print(
            f"[superpoints] Added field '{output_field_name}' "
            f"(type=INSTANCE, points={target_layer.data.points.shape[0]}) on layer '{target_layer.browser_name}'"
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_region_merge(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dlg = RegionMergeDialog(
            scene,
            active_layer,
            settings=self.preferences.get_region_merge_settings(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dlg.selected_layer()
        if layer is None:
            return
        self.preferences.set_region_merge_settings(dlg.settings())
        self.preferences.save()
        source_field_name = dlg.source_field_name()
        if not source_field_name:
            QMessageBox.warning(self, "Merge planar regions", "No source instance field selected.")
            return
        source_fields = layer.data.get_fields(names=source_field_name, field_type=FieldType.INSTANCE)
        if not source_fields:
            QMessageBox.warning(self, "Merge planar regions", "Selected instance field was not found.")
            return

        source_labels = np.asarray(source_fields[0].data, dtype=np.int32)
        if source_labels.ndim == 2:
            if source_labels.shape[1] != 1:
                QMessageBox.warning(
                    self,
                    "Merge planar regions",
                    f"Source field '{source_field_name}' must have shape (N,) or (N,1).",
                )
                return
            source_labels = source_labels[:, 0]
        elif source_labels.ndim != 1:
            QMessageBox.warning(
                self,
                "Merge planar regions",
                f"Source field '{source_field_name}' must have shape (N,) or (N,1).",
            )
            return
        if source_labels.shape[0] != layer.data.points.shape[0]:
            QMessageBox.warning(
                self,
                "Merge planar regions",
                "Source field length does not match the point count.",
            )
            return

        layer_id = layer.id
        output_field_name = self._unique_field_name(
            layer.data.field_names,
            dlg.output_field_base() or "merged_planar_regions",
        )
        progress = _native_region_merge.Progress()

        class _RegionMergeWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, _stats = merge_planar_regions(
                        layer.data,
                        source_labels,
                        params=dlg.params(),
                        progress=progress,
                    )
                    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
                    if labels.shape[0] != layer.data.points.shape[0]:
                        raise RuntimeError("Native output label length does not match point count.")
                    layer.data.add_field(
                        name=output_field_name,
                        data=labels[:, None],
                        field_type=FieldType.INSTANCE,
                    )
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog("Preparing planar region merge...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Merge planar regions")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)

        def _on_cancel() -> None:
            progress.request_cancel()
            progress_dialog.setLabelText("Cancelling...")

        progress_dialog.canceled.connect(_on_cancel)

        timer = QTimer(self)
        timer.setInterval(100)
        t_start = time.perf_counter()

        def _on_tick() -> None:
            try:
                total = progress.total()
                done = progress.done()
                elapsed = time.perf_counter() - t_start
                if total > 0:
                    progress_dialog.setMaximum(int(total))
                    progress_dialog.setValue(min(int(done), int(total)))
                else:
                    progress_dialog.setMaximum(0)
                progress_dialog.setLabelText(
                    f"{progress.stage()} | {elapsed:.1f}s"
                )
            except Exception as exc:  # pragma: no cover - GUI path
                print(f"[region_merge/ui] progress polling failed: {exc}")

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

        worker = _RegionMergeWorker()
        worker.errored.connect(_on_error)
        worker.completed.connect(_on_finished)
        worker.finished.connect(lambda: None)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()

        if error_msg["value"] is not None:
            QMessageBox.critical(self, "Merge planar regions failed", error_msg["value"])
            return
        if progress.cancelled():
            return

        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            target_layer_obj = scene_after.layers.get(layer_id)
            if isinstance(target_layer_obj, PointCloudLayer):
                target_layer = target_layer_obj
                scene_after.active_layer_id = target_layer.id

        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        print(
            f"[region_merge] Added field '{output_field_name}' from source '{source_field_name}' "
            f"(type=INSTANCE, points={target_layer.data.points.shape[0]}) on layer '{target_layer.browser_name}'"
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_connected_components(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dialog = ConnectedComponentsDialog(
            scene,
            active_layer,
            settings=self.preferences.get_connected_components_settings(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dialog.selected_layer()
        if layer is None:
            return
        self.preferences.set_connected_components_settings(dialog.settings())
        self.preferences.save()
        epsilon = dialog.epsilon()

        inverse_mapping: np.ndarray | None = None
        processing_data = layer.data
        if dialog.on_selection_only():
            selection = layer.active_selection
            if selection is None or selection.size == 0:
                QMessageBox.warning(
                    self,
                    "Connected components",
                    "The selected point cloud no longer has an active selection.",
                )
                return
            try:
                processing_data, inverse_mapping = layer.data.extract_subcloud(selection.copy())
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Connected components", str(exc))
                return

        output_mode = dialog.output_mode()
        target_field: InstanceSegmentation | None = None
        if output_mode == "write_existing":
            field_name = dialog.existing_field_name()
            if field_name is None:
                return
            fields = layer.data.get_fields(names=field_name, field_type=FieldType.INSTANCE)
            if not fields or not isinstance(fields[0], InstanceSegmentation):
                QMessageBox.warning(
                    self,
                    "Connected components",
                    "The selected output instance field was not found.",
                )
                return
            target_field = fields[0]
            output_field_name = target_field.name
        else:
            output_field_name = self._unique_field_name(
                layer.data.field_names,
                dialog.output_field_base() or "connected_components",
            )

        result: dict[str, object | None] = {"labels": None, "stats": None}

        class _ConnectedComponentsWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, stats = segment_connected_components(
                        processing_data,
                        epsilon=epsilon,
                    )
                    result["labels"] = labels
                    result["stats"] = stats
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog(
            "Computing connected components...",
            "Cancel",
            0,
            0,
            self,
        )
        progress_dialog.setWindowTitle("Connected components instance segmentation")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        cancelled = {"value": False}

        def _on_cancel() -> None:
            cancelled["value"] = True
            progress_dialog.setLabelText("Finishing cancelled computation...")

        progress_dialog.canceled.connect(_on_cancel)
        loop = QEventLoop(self)
        error_message: dict[str, str | None] = {"value": None}

        def _finish() -> None:
            try:
                progress_dialog.canceled.disconnect(_on_cancel)
            except TypeError:
                pass
            progress_dialog.close()
            loop.quit()

        def _error(message: str) -> None:
            error_message["value"] = message
            _finish()

        worker = _ConnectedComponentsWorker()
        worker.errored.connect(_error)
        worker.completed.connect(_finish)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        loop.exec()
        worker.wait()

        if error_message["value"] is not None:
            QMessageBox.critical(
                self,
                "Connected components failed",
                error_message["value"],
            )
            return
        if cancelled["value"]:
            return
        labels = result["labels"]
        if not isinstance(labels, np.ndarray):
            QMessageBox.critical(
                self,
                "Connected components failed",
                "The computation completed without producing labels.",
            )
            return

        try:
            if target_field is not None:
                target_field.merge_with_segmentation(labels, inverse_mapping)
                output_action = "Updated"
            else:
                if inverse_mapping is None:
                    output_labels = labels
                else:
                    output_labels = np.full(layer.data.points.shape[0], -1, dtype=np.int32)
                    output_labels[inverse_mapping] = labels
                layer.data.add_field(
                    name=output_field_name,
                    data=output_labels[:, None],
                    field_type=FieldType.INSTANCE,
                )
                output_action = "Added"
        except Exception as exc:
            QMessageBox.critical(self, "Connected components failed", str(exc))
            return

        layer_id = layer.id
        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            current_layer = scene_after.layers.get(layer_id)
            if isinstance(current_layer, PointCloudLayer):
                target_layer = current_layer
                scene_after.active_layer_id = target_layer.id
        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        stats = result["stats"] if isinstance(result["stats"], dict) else {}
        component_count = int(stats.get("num_components", 0))
        print(
            f"[connected_components] {output_action} field '{output_field_name}' | "
            f"components={component_count} points={labels.size} epsilon={epsilon:g}"
        )
        self.statusBar().showMessage(
            f"Created {component_count:,} connected component instance(s).",
            8000,
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_corner_cleanup(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
        dlg = CornerCleanupDialog(
            scene,
            active_layer,
            settings=self.preferences.get_corner_cleanup_settings(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        layer = dlg.selected_layer()
        source_field_name = dlg.source_field_name()
        if layer is None or source_field_name is None:
            return
        source_fields = layer.data.get_fields(
            names=source_field_name,
            field_type=FieldType.INSTANCE,
        )
        if not source_fields or not isinstance(source_fields[0], InstanceSegmentation):
            QMessageBox.warning(
                self,
                "Clean up corner regions",
                "The selected source instance field was not found.",
            )
            return
        source_field = source_fields[0]
        source_labels = np.asarray(source_field.data, dtype=np.int32).reshape(-1).copy()
        if source_labels.shape[0] != layer.data.points.shape[0]:
            QMessageBox.warning(
                self,
                "Clean up corner regions",
                "Source field length does not match the point count.",
            )
            return

        self.preferences.set_corner_cleanup_settings(dlg.settings())
        self.preferences.save()
        output_mode = dlg.output_mode()
        output_field_name = (
            source_field_name
            if output_mode == "update_source"
            else self._unique_field_name(
                layer.data.field_names,
                dlg.output_field_base() or "corner_cleaned_regions",
            )
        )
        params = dlg.params()
        processing_data = layer.data
        processing_labels = source_labels
        inverse_mapping: np.ndarray | None = None
        if dlg.on_selection_only():
            selection = layer.active_selection
            if selection is None or selection.size == 0:
                QMessageBox.warning(
                    self,
                    "Clean up corner regions",
                    "The selected point cloud no longer has an active selection.",
                )
                return
            try:
                processing_data, inverse_mapping = layer.data.extract_subcloud(
                    selection.copy(),
                    field_names=[source_field_name],
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Clean up corner regions", str(exc))
                return
            processing_labels = np.asarray(
                processing_data.get_fields(names=source_field_name)[0].data,
                dtype=np.int32,
            ).reshape(-1)

        progress = _native_corner_cleanup.Progress()
        result: dict[str, object | None] = {"labels": None, "stats": None}

        class _CornerCleanupWorker(QThread):
            errored = pyqtSignal(str)
            completed = pyqtSignal()

            def run(self) -> None:
                try:
                    labels, stats = cleanup_corner_regions(
                        processing_data,
                        processing_labels,
                        params=params,
                        progress=progress,
                    )
                    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
                    if labels.shape[0] != processing_data.points.shape[0]:
                        raise RuntimeError("Native output label length does not match point count.")
                    result["labels"] = labels
                    result["stats"] = stats
                    self.completed.emit()
                except Exception as exc:  # pragma: no cover - GUI path
                    self.errored.emit(str(exc))

        progress_dialog = QProgressDialog(
            "Preparing corner cleanup...",
            "Cancel",
            0,
            0,
            self,
        )
        progress_dialog.setWindowTitle("Clean up corner regions")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)

        cancel_requested = {"value": False}

        def _on_cancel() -> None:
            cancel_requested["value"] = True
            progress.request_cancel()
            progress_dialog.setLabelText("Cancelling...")

        progress_dialog.canceled.connect(_on_cancel)
        timer = QTimer(self)
        timer.setInterval(100)
        started = time.perf_counter()

        def _on_tick() -> None:
            try:
                total = progress.total()
                done = progress.done()
                if total > 0:
                    progress_dialog.setMaximum(int(total))
                    progress_dialog.setValue(min(int(done), int(total)))
                else:
                    progress_dialog.setMaximum(0)
                progress_dialog.setLabelText(
                    f"{progress.stage()} | {time.perf_counter() - started:.1f}s"
                )
            except Exception as exc:  # pragma: no cover - GUI path
                print(f"[corner_cleanup/ui] progress polling failed: {exc}")

        timer.timeout.connect(_on_tick)
        loop = QEventLoop(self)
        error_message: dict[str, str | None] = {"value": None}

        def _on_finished() -> None:
            timer.stop()
            try:
                progress_dialog.canceled.disconnect(_on_cancel)
            except TypeError:
                pass
            progress_dialog.close()
            loop.quit()

        def _on_error(message: str) -> None:
            error_message["value"] = message
            _on_finished()

        worker = _CornerCleanupWorker()
        worker.errored.connect(_on_error)
        worker.completed.connect(_on_finished)
        worker.finished.connect(lambda: None)
        progress_dialog.show()
        QApplication.processEvents()
        worker.start()
        timer.start()
        loop.exec()
        worker.wait()

        if error_message["value"] is not None:
            QMessageBox.critical(
                self,
                "Corner cleanup failed",
                error_message["value"],
            )
            return
        if cancel_requested["value"]:
            return
        cleaned_labels = result["labels"]
        if not isinstance(cleaned_labels, np.ndarray):
            QMessageBox.critical(
                self,
                "Corner cleanup failed",
                "Corner cleanup completed without producing labels.",
            )
            return
        stats = result["stats"] if isinstance(result["stats"], dict) else {}
        cleaned_region_count = int(stats.get("num_corner_regions_cleaned", 0))
        cleaned_point_count = int(stats.get("num_corner_points_reassigned", 0))
        if cleaned_region_count == 0:
            candidate_count = int(stats.get("num_corner_candidates", 0))
            destination_count = int(stats.get("num_destination_regions", 0))
            rejected_slenderness = int(stats.get("num_rejected_slenderness", 0))
            rejected_absolute = int(stats.get("num_rejected_absolute_width", 0))
            rejected_relative = int(stats.get("num_rejected_relative_width", 0))
            missing_neighbors = int(stats.get("num_rejected_planar_neighbors", 0))
            insufficient_support = int(stats.get("num_rejected_dual_support", 0))
            radius = float(stats.get("neighbor_radius", 0.0))
            message = (
                "No corner regions met all cleanup criteria.\n\n"
                f"Corner candidates: {candidate_count}\n"
                f"Destination regions: {destination_count}\n"
                f"Rejected by slenderness: {rejected_slenderness}\n"
                f"Rejected by absolute width: {rejected_absolute}\n"
                f"Rejected by relative width: {rejected_relative}\n"
                f"Rejected without two jointly bordering planes: {missing_neighbors}\n"
                f"Rejected by dual-neighbor support: {insufficient_support}\n"
                f"Effective neighbor radius: {radius:g}"
            )
            self.statusBar().showMessage("Corner cleanup made no changes.", 8000)
            QMessageBox.information(self, "Clean up corner regions", message)
            return

        output_labels = source_labels.copy()
        if inverse_mapping is None:
            output_labels = cleaned_labels
        else:
            output_labels[inverse_mapping] = cleaned_labels
        if output_mode == "update_source":
            source_field.data = output_labels.reshape(-1, 1)
            output_action = "Updated"
        else:
            layer.data.add_field(
                name=output_field_name,
                data=output_labels[:, None],
                field_type=FieldType.INSTANCE,
            )
            output_action = "Added"

        layer_id = layer.id
        scene_after = self.scene_manager._scene
        target_layer = layer
        if scene_after is not None:
            target_layer_object = scene_after.layers.get(layer_id)
            if isinstance(target_layer_object, PointCloudLayer):
                target_layer = target_layer_object
                scene_after.active_layer_id = target_layer.id
        target_layer.set_active_field_name(output_field_name)
        target_layer.update()
        self.scene_manager.broadcastActivatedLayer.emit(target_layer)
        self.scene_manager.broadcastActivatedPcdField.emit(target_layer)
        print(
            f"[corner_cleanup] {output_action} field '{output_field_name}' from "
            f"'{source_field_name}' | corners={cleaned_region_count} "
            f"points={cleaned_point_count} candidates={stats.get('num_corner_candidates', 0)} "
            f"destination_regions={stats.get('num_destination_regions', 0)}"
        )
        self.statusBar().showMessage(
            f"Cleaned {cleaned_region_count} corner region(s), "
            f"reassigning {cleaned_point_count:,} point(s).",
            8000,
        )
        dataset = self.dataset_manager._dataset
        if dataset is not None and scene_after is not None:
            for ref in dataset.doc_refs:
                if ref.name == scene_after.doc.name:
                    ref.modState = RefModState.MODIFIED
                    break
        self.scene_manager.populate_tree()
        self.viewer.rerender()

    def _on_compute_geometric_features(self) -> None:
        scene = self.scene_manager._scene
        active_layer = require_active_layer(self, scene, PointCloudLayer)
        if active_layer is None:
            return
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
                        optional_feature_field_names=dlg.optional_feature_field_names(),
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
 
