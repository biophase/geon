import time
import math
from enum import Enum

import numpy as np
from plyfile import PlyData
from PyQt6.QtCore import Qt, QSize, QPoint, QTimer
from PyQt6.QtGui import QIcon, QPalette, QColor
from PyQt6.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout,
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
        # Only intercept for lasso if active and not paused.
        if self.viewer.lasso_active and not self.viewer.lasso_paused:
            self.viewer.lasso_click_event(obj, event)
            return

        current_time = time.time()
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
        self.viewer.zoom_camera(1.0/1.1)

    def double_click_event(self):
        """Use vtkPointPicker to select a point and update the pivot."""
        x, y = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
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
        # A new "Deselect" button is added at the end.
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
            ("resources/lasso.png", "Performs lasso segmentation", self.lasso_segmentation),
            ("resources/deselect.png", "Deselect active selection", self.deselect_selection)
        ]

        # Populate the grid layout with 2 columns.
        row = 0
        col = 0
        for item in self.tool_list:
            if isinstance(item, ToolBarOptions):
                if col != 0:
                    row += 1
                    col = 0
                if item == ToolBarOptions.BREAK:
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
                icon_path, tooltip, func = item
                btn = QPushButton()
                btn.setIcon(QIcon(icon_path))
                btn.setToolTip(tooltip)
                btn.setIconSize(QSize(28, 28))
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
        self.pivot_sphere_source = None
        self.pivot_actor = None
        self.point_cloud_actor = None

        # Selection management
        self.active_selection = None  # NumPy array of indices (or None if no selection)
        self.selection_actor = None   # Actor for the active selection (pulsing)
        self.selection_timer = QTimer()
        self.selection_timer.timeout.connect(self.update_selection_pulse)
        self.selection_phase = 0.0

        # Variables for lasso segmentation:
        self.lasso_active = False
        self.lasso_paused = False
        self.lasso_points = []  # List of screen-space vertices
        self.lasso_candidate_indices = None  # Candidates from lasso tool
        self.lasso_actor = None           # Actor for candidate points during lasso
        self.lasso_polygon_actor = None   # Actor for the lasso polygon (and ghost line)
        self.lasso_control_panel = None   # Floating control panel widget
        self.lasso_click_observer = None  # Observer id for lasso left-clicks
        self.lasso_mouse_move_observer = None  # Observer for lasso mouse moves
        self.lasso_key_observer = None    # Observer for lasso key events

        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()

    def zoom_camera(self, factor):
        camera = self.renderer.GetActiveCamera()
        camera_pos = np.array(camera.GetPosition())
        focal_point = np.array(self.pivot_point)
        direction = focal_point - camera_pos
        new_position = camera_pos + direction * (1 - factor)
        camera.SetPosition(*new_position)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def toggle_projection(self):
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        if file_path:
            print(f"Loading: {file_path}")
            ply_data = PlyData.read(file_path)
            self.display_ply(ply_data)

    def display_ply(self, ply_data):
        vertex_data = ply_data['vertex']
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        initial_pivot = np.mean(points, axis=0).tolist()
        has_color = all(c in vertex_data.data.dtype.names for c in ('red', 'green', 'blue'))
        if has_color:
            colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
        else:
            colors = np.full((len(points), 3), [1.0, 1.0, 1.0])
        self.all_points = points.copy()  # Save for segmentation
        self.all_colors = colors.copy()

        # --- Full Resolution PolyData ---
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

        # --- Low Resolution PolyData (LOD) ---
        # Subsample points and colors (every 100th point)
        lod_points = points[::100, ...]
        lod_colors = colors[::100, ...]
        points_vtk_lod0 = ns.numpy_to_vtk(lod_points, deep=False)
        colors_vtk_lod0 = ns.numpy_to_vtk(lod_colors.astype(np.uint8), deep=False)
        vtk_points_lod0 = vtk.vtkPoints()
        vtk_points_lod0.SetData(points_vtk_lod0)
        poly_data_lod0 = vtk.vtkPolyData()
        poly_data_lod0.SetPoints(vtk_points_lod0)
        poly_data_lod0.GetPointData().SetScalars(colors_vtk_lod0)
        vertex_lod0 = vtk.vtkVertexGlyphFilter()
        vertex_lod0.SetInputData(poly_data_lod0)
        mapper_lod0 = vtk.vtkPolyDataMapper()
        mapper_lod0.SetInputConnection(vertex_lod0.GetOutputPort())
        mapper_lod0.SetScalarModeToUsePointData()
        mapper_lod0.Update()

        # --- Create the LOD Actor ---
        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.AddLODMapper(mapper_lod0)
        actor.GetProperty().SetPointSize(2)  # Adjust point size for clarity

        self.point_cloud_actor = actor
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.set_pivot_point(initial_pivot)
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()


    def update_pivot_visualization(self):
        if self.pivot_sphere_source is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(0.05)
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
        self.pivot_point = list(new_pivot)
        self.renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self.renderer.ResetCameraClippingRange()
        self.update_pivot_visualization()
        self.vtkWidget.GetRenderWindow().Render()

    def update_interactor(self):
        interactor_style = InteractorStyle(self.renderer, self)
        self.interactor.SetInteractorStyle(interactor_style)

    # --- Active Selection Management ---
    def update_selection_visualization(self):
        """If an active selection exists, update the visualization:
           - Set the main point cloud opacity to 15%
           - Create/update a separate actor for the selected points, pulsating in color.
        """
        if self.active_selection is not None and self.active_selection.size > 0:
            # Lower the overall opacity
            # self.point_cloud_actor.GetProperty().SetOpacity(0.15)
            # Build polydata for selected points
            selected_points = self.all_points[self.active_selection]
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(ns.numpy_to_vtk(selected_points, deep=True))
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)
            # Use a vertex glyph filter for points
            vertex = vtk.vtkVertexGlyphFilter()
            vertex.SetInputData(poly_data)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex.GetOutputPort())
            mapper.ScalarVisibilityOff()  # Use property color instead of scalars
            if self.selection_actor is None:
                # Use a standard vtkActor to allow dynamic property updates.
                self.selection_actor = vtk.vtkActor()
                self.selection_actor.GetProperty().SetPointSize(self.point_cloud_actor.GetProperty().GetPointSize() + 2)
                self.selection_actor.GetProperty().SetColor((1., 0.5, 0.))  # Red color for selection
                # self.selection_actor.GetProperty().SetOpacity(1.0)
                self.renderer.AddActor(self.selection_actor)
            self.selection_actor.SetMapper(mapper)
            # Start the pulsating timer if not already running
            if not self.selection_timer.isActive():
                self.selection_phase = 0.0
                self.selection_timer.start(50)
        else:
            # No active selection: restore full opacity and remove selection actor
            # self.point_cloud_actor.GetProperty().SetOpacity(1.0)
            if self.selection_actor is not None:
                self.renderer.RemoveActor(self.selection_actor)
                self.selection_actor = None
            self.selection_timer.stop()
        self.vtkWidget.GetRenderWindow().Render()

    def update_selection_pulse(self):
        """Timer callback: update the color of the selection actor in a sine wave pattern."""
        if self.selection_actor is None:
            return
        # Increase phase and compute new red intensity between 0.25 and 1.0
        self.selection_phase += 0.1
        new_red = 0.25 + 0.75 * ((math.sin(self.selection_phase*4) + 1) / 2)
        self.selection_actor.GetProperty().SetColor(new_red, new_red*0.5, 0.0)
        self.vtkWidget.GetRenderWindow().Render()

    def deselect_selection(self):
        """Clear any active selection."""
        self.active_selection = None
        # Restore full opacity for main actor
        self.point_cloud_actor.GetProperty().SetOpacity(1.0)
        if self.selection_actor is not None:
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None
        self.selection_timer.stop()
        print("Active selection cleared.")
        self.vtkWidget.GetRenderWindow().Render()


    def compute_display_coordinates(self, points):
        """
        Transform an array of world-space points (Nx3) to display-space (Nx2) using
        the renderer's active camera projection. This avoids per-point transformation calls.
        """
        camera = self.renderer.GetActiveCamera()
        # Get render window size from VTK
        window_size = self.vtkWidget.GetRenderWindow().GetSize()  # returns (width, height)
        width, height = window_size
        clipping_range = camera.GetClippingRange()
        aspect = self.renderer.GetTiledAspectRatio()
        # Get the composite projection transform matrix
        matrix = camera.GetCompositeProjectionTransformMatrix(aspect, clipping_range[0], clipping_range[1])
        m = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                m[i, j] = matrix.GetElement(i, j)
        N = points.shape[0]
        # Convert points to homogeneous coordinates
        pts_hom = np.hstack((points, np.ones((N, 1))))
        pts_clip = pts_hom @ m.T
        # Perspective division to get normalized device coordinates (range [-1, 1])
        pts_ndc = pts_clip[:, :3] / pts_clip[:, 3:4]
        # Map NDC to display coordinates
        display_x = (pts_ndc[:, 0] + 1) / 2 * width
        display_y = (pts_ndc[:, 1] + 1) / 2 * height
        return np.column_stack((display_x, display_y))


    
    # --- Lasso Segmentation Methods ---
    @staticmethod
    def vectorized_point_in_poly(points, poly):
        """
        Vectorized point-in-polygon test.
        
        :param points: Nx2 array of candidate display coordinates.
        :param poly: Mx2 array of polygon vertices.
        :return: Boolean array of length N; True for points inside the polygon.
        """
        poly = np.asarray(poly)
        x = points[:, 0]
        y = points[:, 1]
        poly_x = poly[:, 0]
        poly_y = poly[:, 1]
        # Roll the polygon coordinates to get (xi,yi) and (xj,yj) pairs for each edge
        poly_x_next = np.roll(poly_x, -1)
        poly_y_next = np.roll(poly_y, -1)
        # Compute the intersection condition for each candidate point against all edges
        cond = ((poly_y > y[:, None]) != (poly_y_next > y[:, None])) & (
            x[:, None] < (poly_x_next - poly_x) * (y[:, None] - poly_y) / (poly_y_next - poly_y + 1e-12) + poly_x
        )
        inside = np.count_nonzero(cond, axis=1) % 2 == 1
        return inside
    
    def lasso_segmentation(self):
        """Activate lasso segmentation mode."""
        if not hasattr(self, "all_points"):
            QMessageBox.warning(self, "Lasso Segmentation", "No point cloud loaded!")
            return
        print("Lasso segmentation activated.")
        self.lasso_active = True
        self.lasso_paused = False
        self.lasso_points = []
        self.lasso_candidate_indices = np.arange(len(self.all_points))
        self.point_cloud_actor.VisibilityOff()
        self.create_lasso_actor()
        self.interactor.SetInteractorStyle(None)
        self.lasso_control_panel = QDialog(self)
        self.lasso_control_panel.setWindowTitle("Lasso Controls")
        self.lasso_control_panel.setWindowFlags(self.lasso_control_panel.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        panel_layout = QHBoxLayout(self.lasso_control_panel)
        btn_pause = QPushButton("Pause")
        btn_pause.clicked.connect(self.lasso_toggle_pause)
        btn_escape = QPushButton("Escape")
        btn_escape.clicked.connect(self.lasso_escape)
        btn_in = QPushButton("In")
        btn_in.clicked.connect(self.lasso_in)
        btn_out = QPushButton("Out")
        btn_out.clicked.connect(self.lasso_out)
        btn_accept = QPushButton("Accept")
        btn_accept.clicked.connect(self.lasso_accept)
        btn_reject = QPushButton("Reject")
        btn_reject.clicked.connect(self.lasso_reject)
        for btn in [btn_pause, btn_escape, btn_in, btn_out, btn_accept, btn_reject]:
            panel_layout.addWidget(btn)
        self.lasso_control_panel.setLayout(panel_layout)
        pos = self.vtkWidget.mapToGlobal(self.vtkWidget.rect().topLeft())
        self.lasso_control_panel.move(pos.x() + 10, pos.y() + 10)
        self.lasso_control_panel.show()
        self.lasso_click_observer = self.interactor.AddObserver("LeftButtonPressEvent", self.lasso_click_event)
        self.lasso_mouse_move_observer = self.interactor.AddObserver("MouseMoveEvent", self.lasso_mouse_move_event)
        self.lasso_key_observer = self.interactor.AddObserver("KeyPressEvent", self.lasso_key_event)

    def lasso_click_event(self, obj, event):
        if not self.lasso_active or self.lasso_paused:
            return
        pos = self.interactor.GetEventPosition()
        self.lasso_points.append(pos)
        self.update_lasso_polygon_actor()

    def lasso_mouse_move_event(self, obj, event):
        if self.lasso_active and not self.lasso_paused and len(self.lasso_points) > 0:
            ghost = self.interactor.GetEventPosition()
            self.update_lasso_polygon_actor(ghost_point=ghost)
        return

    def update_lasso_polygon_actor(self, ghost_point=None):
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        points_list = list(self.lasso_points)
        if ghost_point is not None:
            points_list.append(ghost_point)
        if len(points_list) < 2:
            self.vtkWidget.GetRenderWindow().Render()
            return
        points = vtk.vtkPoints()
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(points_list))
        for i, (x, y) in enumerate(points_list):
            points.InsertNextPoint(x, y, 0)
            polyLine.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0)  # Orange
        actor.GetProperty().SetLineWidth(4)
        self.lasso_polygon_actor = actor
        self.renderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def create_lasso_actor(self):
        candidate_points = self.all_points[self.lasso_candidate_indices]
        candidate_colors = self.all_colors[self.lasso_candidate_indices]
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(candidate_points, deep=True))
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        colors_vtk = ns.numpy_to_vtk(candidate_colors.astype(np.uint8), deep=True)
        poly_data.GetPointData().SetScalars(colors_vtk)
        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        if self.lasso_actor is None:
            self.lasso_actor = vtk.vtkLODActor()
            self.lasso_actor.GetProperty().SetPointSize(self.point_cloud_actor.GetProperty().GetPointSize())
            self.renderer.AddActor(self.lasso_actor)
        self.lasso_actor.SetMapper(mapper)
        self.vtkWidget.GetRenderWindow().Render()

    def lasso_toggle_pause(self):
        self.lasso_paused = not self.lasso_paused
        if self.lasso_paused:
            # Enable camera interaction.
            self.interactor.SetInteractorStyle(InteractorStyle(self.renderer, self))
            self.lasso_points = []
            if self.lasso_polygon_actor is not None:
                self.renderer.RemoveActor(self.lasso_polygon_actor)
                self.lasso_polygon_actor = None
            print("Lasso segmentation paused.")
        else:
            self.interactor.SetInteractorStyle(None)
            print("Lasso segmentation resumed.")
        self.vtkWidget.GetRenderWindow().Render()

    def lasso_escape(self):
        print("Lasso polygon cleared (escape).")
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        self.vtkWidget.GetRenderWindow().Render()

    def point_in_polygon(self, x, y, poly):
        num = len(poly)
        j = num - 1
        c = False
        for i in range(num):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                c = not c
            j = i
        return c

    def lasso_in(self):
        if len(self.lasso_points) < 3:
            print("Lasso polygon not complete (need at least 3 points).")
            return
        poly = self.lasso_points  # List of (x, y) screen-space vertices
        candidate_indices = self.lasso_candidate_indices
        candidate_points = self.all_points[candidate_indices]  # (N x 3) world-space points

        # Compute display coordinates in a vectorized way
        display_coords = self.compute_display_coordinates(candidate_points)

        # Pre-culling: filter out points outside the bounding box of the lasso polygon
        lasso_poly = np.array(poly)
        min_x, max_x = lasso_poly[:, 0].min(), lasso_poly[:, 0].max()
        min_y, max_y = lasso_poly[:, 1].min(), lasso_poly[:, 1].max()
        bbox_mask = (
            (display_coords[:, 0] >= min_x) & (display_coords[:, 0] <= max_x) &
            (display_coords[:, 1] >= min_y) & (display_coords[:, 1] <= max_y)
        )
        indices_in_bbox = np.nonzero(bbox_mask)[0]
        filtered_display_coords = display_coords[indices_in_bbox]

        # Run the vectorized point-in-polygon test on the culled set
        inside_mask = self.vectorized_point_in_poly(filtered_display_coords, lasso_poly)
        # Map the indices from the culled array back to the overall candidate indices
        selected_in_bbox = indices_in_bbox[inside_mask]
        new_candidates = candidate_indices[selected_in_bbox]

        self.lasso_candidate_indices = new_candidates
        print(f"Lasso 'in': {len(self.lasso_candidate_indices)} candidates remain.")
        self.create_lasso_actor()
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None

    def lasso_out(self):
        if len(self.lasso_points) < 3:
            print("Lasso polygon not complete (need at least 3 points).")
            return
        poly = self.lasso_points  # List of (x, y) screen-space vertices
        candidate_indices = self.lasso_candidate_indices
        candidate_points = self.all_points[candidate_indices]  # (N x 3) world-space points

        # Compute display coordinates in a vectorized way
        display_coords = self.compute_display_coordinates(candidate_points)

        # Pre-culling: filter out points outside the bounding box of the lasso polygon
        lasso_poly = np.array(poly)
        min_x, max_x = lasso_poly[:, 0].min(), lasso_poly[:, 0].max()
        min_y, max_y = lasso_poly[:, 1].min(), lasso_poly[:, 1].max()
        bbox_mask = (
            (display_coords[:, 0] >= min_x) & (display_coords[:, 0] <= max_x) &
            (display_coords[:, 1] >= min_y) & (display_coords[:, 1] <= max_y)
        )
        indices_in_bbox = np.nonzero(bbox_mask)[0]
        filtered_display_coords = display_coords[indices_in_bbox]

        # Run the vectorized point-in-polygon test on the culled set
        inside_mask = self.vectorized_point_in_poly(filtered_display_coords, lasso_poly)
        # For 'lasso out', we want points outside the polygon
        outside_mask = ~inside_mask
        selected_in_bbox = indices_in_bbox[outside_mask]
        new_candidates = candidate_indices[selected_in_bbox]

        self.lasso_candidate_indices = new_candidates
        print(f"Lasso 'out': {len(self.lasso_candidate_indices)} candidates remain.")
        self.create_lasso_actor()
        self.lasso_points = []
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None

    def lasso_accept(self):
        print("Lasso selection accepted.")
        new_sel = self.lasso_candidate_indices.copy()
        print("New selection:", new_sel)
        # If an active selection already exists, ask how to combine
        if self.active_selection is not None and self.active_selection.size > 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Combine Selections")
            msg.setText("An active selection already exists. How would you like to combine it with the new selection?")
            union_btn = msg.addButton("Union", QMessageBox.ButtonRole.AcceptRole)
            intersect_btn = msg.addButton("Intersection", QMessageBox.ButtonRole.AcceptRole)
            diff_btn = msg.addButton("Difference (new removed from old)", QMessageBox.ButtonRole.AcceptRole)
            rev_diff_btn = msg.addButton("Reverse Difference (old removed from new)", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            if msg.clickedButton() == union_btn:
                combined = np.union1d(self.active_selection, new_sel)
            elif msg.clickedButton() == intersect_btn:
                combined = np.intersect1d(self.active_selection, new_sel)
            elif msg.clickedButton() == diff_btn:
                combined = np.setdiff1d(self.active_selection, new_sel)
            elif msg.clickedButton() == rev_diff_btn:
                combined = np.setdiff1d(new_sel, self.active_selection)
            else:
                print("Combination cancelled; keeping previous selection.")
                combined = self.active_selection
            self.active_selection = combined
        else:
            self.active_selection = new_sel
        self.exit_lasso_mode()
        self.update_selection_visualization()

    def lasso_reject(self):
        print("Lasso selection rejected.")
        self.exit_lasso_mode()

    def lasso_key_event(self, obj, event):
        key = self.interactor.GetKeySym()
        if key == "space":
            self.lasso_toggle_pause()
        elif key == "Escape":
            self.lasso_escape()
        elif key.lower() == "i":
            self.lasso_in()
        elif key.lower() == "o":
            self.lasso_out()
        elif key == "Return":
            self.lasso_accept()
        return

    def exit_lasso_mode(self):
        self.lasso_active = False
        if self.lasso_click_observer is not None:
            self.interactor.RemoveObserver(self.lasso_click_observer)
            self.lasso_click_observer = None
        if self.lasso_mouse_move_observer is not None:
            self.interactor.RemoveObserver(self.lasso_mouse_move_observer)
            self.lasso_mouse_move_observer = None
        if self.lasso_key_observer is not None:
            self.interactor.RemoveObserver(self.lasso_key_observer)
            self.lasso_key_observer = None
        if self.lasso_polygon_actor is not None:
            self.renderer.RemoveActor(self.lasso_polygon_actor)
            self.lasso_polygon_actor = None
        if self.lasso_actor is not None:
            self.renderer.RemoveActor(self.lasso_actor)
            self.lasso_actor = None
        if self.lasso_control_panel is not None:
            self.lasso_control_panel.close()
            self.lasso_control_panel = None
        self.point_cloud_actor.VisibilityOn()
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()

    # --- Tool Button Functions ---
    def deselect_selection(self):
        """Clear any active selection."""
        self.active_selection = None
        self.point_cloud_actor.GetProperty().SetOpacity(1.0)
        if self.selection_actor is not None:
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None
        self.selection_timer.stop()
        print("Active selection cleared.")
        self.vtkWidget.GetRenderWindow().Render()

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
