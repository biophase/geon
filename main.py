import time
from enum import Enum

import numpy as np
from plyfile import PlyData
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPalette, QColor
from PyQt6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
                             QMainWindow, QMessageBox, QPushButton, QSpacerItem,
                             QSizePolicy, QWidget, QFileDialog)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support as ns
import vtk

from theme import set_dark_palette


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Custom interactor style using vtkPointPicker for fast pivot picking."""
    
    def __init__(self, renderer, viewer):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.camera = renderer.GetActiveCamera()
        self.last_click_time = 0  # For double-click detection

        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)

    def left_button_press_event(self, obj, event):
        current_time = time.time()
        # Use a 300ms threshold for double-click detection
        if current_time - self.last_click_time < 0.3:
            self.double_click_event()
        else:
            self.last_click_time = current_time
            self.OnLeftButtonDown()

    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()

    def mouse_move_event(self, obj, event):
        self.OnMouseMove()

    def mouse_wheel_forward_event(self, obj, event):
        self.viewer.zoom_camera(1.1)

    def mouse_wheel_backward_event(self, obj, event):
        self.viewer.zoom_camera(0.9)

    def double_click_event(self):
        """Use vtkPointPicker to select a point and update the pivot."""
        x, y = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)  # Adjust tolerance as needed
        picker.Pick(x, y, 0, self.renderer)
        if picker.GetPointId() < 0:
            print("No valid point picked.")
            return
        picked_point = picker.GetPickPosition()
        print(f"New pivot set at: {picked_point}")
        self.viewer.set_pivot_point(picked_point)

    def key_press_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "F3":
            self.viewer.toggle_projection()
        self.OnKeyPress()


class ToolBarOptions(Enum):
    BREAK = 1
    PLACEHOLDER = 2


class VTKPointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK PLY Point Cloud Viewer")
        self.resize(800, 600)

        # Main container and horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        h_layout = QHBoxLayout(central_widget)

        # --- VTK Renderer Widget (left) ---
        self.vtkWidget = QVTKRenderWindowInteractor(central_widget)
        h_layout.addWidget(self.vtkWidget, stretch=1)

        # --- Tool Panel (right) ---
        tool_panel_container = QWidget()
        tool_panel_container.setFixedWidth(120)
        tool_panel_layout = QGridLayout(tool_panel_container)
        tool_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        h_layout.addWidget(tool_panel_container)

        # Define the tool list.
        # BREAK inserts a horizontal separator,
        # PLACEHOLDER inserts an expanding spacer.
        self.tool_list = [
            ToolBarOptions.BREAK,  # File I/O and visualization group
            ("resources/load.png", "Load PointCloud", self.load_ply_file),
            ("resources/settings.png", "Settings", self.settings),
            ("resources/increase_point_size.png", "Increase point size", self.increase_point_size),
            ("resources/decrease_point_size.png", "Decrease point size", self.decrease_point_size),
            ("resources/reset_view.png", "Reset View", self.reset_view),
            ToolBarOptions.BREAK,  # Segmentation tools group
            ("resources/load_segmentation.png", "Load segmentation", self.load_segmentation),
            ("resources/region_growing.png", "Perform region growing", self.region_growing),
            ("resources/classify_selection.png", "Classify selection", self.classify_selection),
            ("resources/semantic_class_manager.png", "Edit semantic schema", self.semantic_class_manager),
            ("resources/load_semantic_class_schema.png", "Load semantic schema", self.load_semantic_class_schema),
        ]

        # Populate the grid layout with 2 columns.
        row = 0
        col = 0
        for item in self.tool_list:
            if isinstance(item, ToolBarOptions):
                # If we're mid-row, start a new row before inserting a full-width item.
                if col != 0:
                    row += 1
                    col = 0
                if item == ToolBarOptions.BREAK:
                    # Create a horizontal line separator spanning 2 columns.
                    line = QFrame()
                    line.setFrameShape(QFrame.Shape.HLine)
                    line.setFrameShadow(QFrame.Shadow.Sunken)
                    tool_panel_layout.addWidget(line, row, 0, 1, 2)
                    row += 1
                elif item == ToolBarOptions.PLACEHOLDER:
                    spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
                    tool_panel_layout.addItem(spacer, row, 0, 1, 2)
                    row += 1
            else:
                # Regular tool: tuple (icon_path, tooltip, function)
                icon_path, tooltip, func = item
                btn = QPushButton()
                btn.setIcon(QIcon(icon_path))
                btn.setToolTip(tooltip)
                btn.setIconSize(QSize(28, 28))  # Increase icon size to 64x64
                btn.clicked.connect(func)
                tool_panel_layout.addWidget(btn, row, col)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1

        # --- End of Tool Panel Setup ---

        # VTK Setup
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        # Initial pivot at [0, 0, 0]
        self.pivot_point = [0, 0, 0]

        # Pivot visualization: a semi-transparent purple sphere (10cm diameter)
        self.pivot_sphere_source = None
        self.pivot_actor = None

        # Save the point cloud actor for point size modifications.
        self.point_cloud_actor = None

        self.update_interactor()

    def zoom_camera(self, factor):
        """Zooms the camera while keeping the pivot fixed."""
        camera = self.renderer.GetActiveCamera()
        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(self.pivot_point)
        direction = focal_point - camera_pos
        new_position = camera_pos + direction * (1 - factor)
        camera.SetPosition(*new_position)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def toggle_projection(self):
        """Toggles between perspective and parallel projection (F3)."""
        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            print("Switching to Perspective Projection")
            camera.SetParallelProjection(False)
        else:
            print("Switching to Parallel Projection")
            camera.SetParallelProjection(True)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def load_ply_file(self):
        """Load a PLY file and display the point cloud with colors."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        if file_path:
            print(f"Loading: {file_path}")
            ply_data = PlyData.read(file_path)
            self.display_ply(ply_data)

    def display_ply(self, ply_data):
        """Fast PLY Loading with Colors."""
        vertex_data = ply_data['vertex']
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        # Use the point cloud's mean as the initial pivot.
        initial_pivot = np.mean(points, axis=0).tolist()

        has_color = all(c in vertex_data.data.dtype.names for c in ('red', 'green', 'blue'))
        if has_color:
            colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
        else:
            colors = np.full((len(points), 3), [1.0, 1.0, 1.0])  # default to white

        vtk_points = vtk.vtkPoints()
        points_vtk = ns.numpy_to_vtk(points, deep=False)
        colors_vtk = ns.numpy_to_vtk(colors.astype(np.uint8), deep=False)
        vtk_points.SetData(points_vtk)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.GetPointData().SetScalars(colors_vtk)

        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.Update()

        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(3)
        self.point_cloud_actor = actor

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        self.set_pivot_point(initial_pivot)
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()

    def update_pivot_visualization(self):
        """Creates or updates a semi-transparent purple sphere at the current pivot."""
        if self.pivot_sphere_source is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(0.05)  # 5cm radius (10cm diameter)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            sphere.SetCenter(*self.pivot_point)
            self.pivot_sphere_source = sphere

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            self.pivot_actor = vtk.vtkActor()
            self.pivot_actor.SetMapper(mapper)
            self.pivot_actor.GetProperty().SetColor(0.5, 0, 0.5)
            self.pivot_actor.GetProperty().SetOpacity(0.5)
            self.pivot_actor.GetProperty().LightingOff()

            self.renderer.AddActor(self.pivot_actor)
        else:
            self.pivot_sphere_source.SetCenter(*self.pivot_point)

    def set_pivot_point(self, new_pivot):
        """Updates the pivot point and its visualization."""
        self.pivot_point = list(new_pivot)
        self.renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self.renderer.ResetCameraClippingRange()
        self.update_pivot_visualization()
        self.vtkWidget.GetRenderWindow().Render()

    def update_interactor(self):
        interactor_style = InteractorStyle(self.renderer, self)
        self.interactor.SetInteractorStyle(interactor_style)

    # --- Tool Button Functions ---

    # File I/O and Visualization Tools:
    def settings(self):
        QMessageBox.information(self, "Settings", "Settings window (dummy).")

    def increase_point_size(self):
        if self.point_cloud_actor:
            current_size = self.point_cloud_actor.GetProperty().GetPointSize()
            new_size = current_size + 1
            print(f"Increasing point size: {current_size} -> {new_size}")
            self.point_cloud_actor.GetProperty().SetPointSize(new_size)
            self.vtkWidget.GetRenderWindow().Render()

    def decrease_point_size(self):
        if self.point_cloud_actor:
            current_size = self.point_cloud_actor.GetProperty().GetPointSize()
            new_size = max(1, current_size - 1)
            print(f"Decreasing point size: {current_size} -> {new_size}")
            self.point_cloud_actor.GetProperty().SetPointSize(new_size)
            self.vtkWidget.GetRenderWindow().Render()

    def reset_view(self):
        self.renderer.ResetCamera()
        print("Reset view: pivot centered on point cloud (dummy implementation).")
        self.vtkWidget.GetRenderWindow().Render()

    # Segmentation Tools (Dummy Implementations):
    def load_segmentation(self):
        print("load_segmentation: Not implemented.")
        QMessageBox.information(self, "Segmentation", "load_segmentation is not implemented yet.")

    def region_growing(self):
        print("region_growing: Not implemented.")
        QMessageBox.information(self, "Segmentation", "region_growing is not implemented yet.")

    def classify_selection(self):
        print("classify_selection: Not implemented.")
        QMessageBox.information(self, "Segmentation", "classify_selection is not implemented yet.")

    def semantic_class_manager(self):
        print("semantic_class_manager: Not implemented.")
        QMessageBox.information(self, "Segmentation", "semantic_class_manager is not implemented yet.")

    def load_semantic_class_schema(self):
        print("load_semantic_class_schema: Not implemented.")
        QMessageBox.information(self, "Segmentation", "load_semantic_class_schema is not implemented yet.")


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    set_dark_palette(app)
    window = VTKPointCloudViewer()
    window.show()
    app.exec()
