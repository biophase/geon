from .viewer import TreeDock, VTKViewer
from .common_tools import CommonToolsDock
from .menu_bar import MenuBar
from .context_ribbon import ContextRibbon
from .field_mapping_dialog import FieldMappingDialog
from geon.io.ply import ply_to_pcd


from geon.rendering.scene import Scene
from PyQt6.QtWidgets import (QMainWindow, QDialog, QFileDialog)
from PyQt6.QtCore import Qt


from plyfile import PlyData
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("geon")
        self.resize(1200,800)

        # content
        self.active_scene: Scene = Scene()

        # settings
        self.setDockOptions(
                QMainWindow.DockOption.AllowTabbedDocks
            |   QMainWindow.DockOption.AllowNestedDocks
            |   QMainWindow.DockOption.GroupedDragging
        )        
        
        # widget initialization
        self.ribbon = ContextRibbon(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.ribbon)
        self.scene_widget = TreeDock("Scene", self) # TODO: might need to specialize the tree class
        self.dataset_widget = TreeDock("Dataset", self) # TODO: might need to specialize the tree class
        self.setMenuBar(MenuBar(self))
        
        self.viewer = VTKViewer(self)
        self.setCentralWidget(self.viewer)

        self.tool_dock = CommonToolsDock("Tools", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.tool_dock)


        
        # initial float widget placement
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dataset_widget)
        self.tabifyDockWidget(self.scene_widget, self.dataset_widget)


        ... # TODO: implement state, scene, undo manager, ui, tools, etc.   


        # temp test ply field dialog
        self.load_ply_and_map_fields()

    def load_ply_and_map_fields(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from plyfile import PlyData
        import numpy as np

        from geon.ui.field_mapping_dialog import FieldMappingDialog
        

        # --- choose file ---
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PLY",
            "",
            "PLY files (*.ply)"
        )
        if not path:
            return

        # --- load PLY ---
        try:
            ply = PlyData.read(path)
        except Exception as e:
            QMessageBox.critical(self, "PLY Error", f"Failed to read PLY:\n{e}")
            return

        if "vertex" not in ply:
            QMessageBox.critical(self, "PLY Error", "PLY has no 'vertex' element.")
            return

        vertex = ply["vertex"]
        names = vertex.data.dtype.names or ()

        # --- detected fields excluding XYZ ---
        detected_fields: list[tuple[str, str]] = []
        for name in names:
            if name in ("x", "y", "z"):
                continue
            dt = np.dtype(vertex[name].dtype)
            detected_fields.append((name, str(dt)))

        # --- show field mapping dialog ---
        dlg = FieldMappingDialog(self, detected_fields)
        result = dlg.exec()  # for PyQt6
        if result != dlg.DialogCode.Accepted:
            return

        # --- retrieve graph-defined mapping ---
        fields_map, field_types = dlg.get_mappings()

        print("FIELD MAP →", fields_map)
        print("FIELD TYPES →", field_types)

        # --- convert PLY to PointCloudData using your existing helper ---
        try:
            pcd = ply_to_pcd(path, fields_map=fields_map, field_types=field_types)
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"Failed to create PointCloudData:\n{e}")
            return

        # --- add to scene/document ---
        layer = self.active_scene.add_data(pcd)
        self.statusBar().showMessage(f"Loaded {path} into layer {layer.id}", 4000)

