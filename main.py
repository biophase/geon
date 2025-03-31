import time
from scipy.spatial import KDTree
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support as ns
import vtk
from plyfile import PlyData
import numpy as np

class CloudCompareStyleInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """Custom interactor style for CloudCompare-like camera navigation with fast pivot picking."""
    
    def __init__(self, renderer, viewer):
        super().__init__()
        self.viewer = viewer  # Reference to main viewer
        self.renderer = renderer
        self.camera = renderer.GetActiveCamera()
        
        self.last_click_time = 0  # To track double-click manually

        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)

    def left_button_press_event(self, obj, event):
        """Manually detect a double-click and find pivot efficiently"""
        current_time = time.time()
        if current_time - self.last_click_time < 0.3:  # 300ms threshold for double-click
            self.double_click_event()
        else:
            self.last_click_time = current_time
            self.OnLeftButtonDown()

    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()

    def mouse_move_event(self, obj, event):
        self.OnMouseMove()

    def mouse_wheel_forward_event(self, obj, event):
        """Zoom in while keeping the pivot fixed."""
        self.viewer.zoom_camera(1.1)
        # self.OnMouseWheelForward()

    def mouse_wheel_backward_event(self, obj, event):
        """Zoom out while keeping the pivot fixed."""
        self.viewer.zoom_camera(0.9)
        # self.OnMouseWheelBackward()

    def double_click_event(self):
        """Find pivot efficiently using depth buffer and KDTree."""
        x, y = self.GetInteractor().GetEventPosition()
        depth = self.viewer.get_depth_at_pixel(x, y)

        if depth is None:
            print("No valid depth value found.")
            return

        world_point = self.viewer.unproject(x, y, depth)
        nearest_point = self.viewer.get_nearest_point(world_point)

        if nearest_point is not None:
            print(f"New pivot set at: {nearest_point}")
            self.viewer.set_pivot_point(nearest_point)
    def key_press_event(self, obj, event):
        """Handles key press events like F3 for toggling projection mode."""
        key = self.GetInteractor().GetKeySym()
        if key == "F3":
            self.viewer.toggle_projection()
        self.OnKeyPress()



class VTKPointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK PLY Point Cloud Viewer")
        self.resize(800, 600)

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout()
        self.centralWidget.setLayout(layout)

        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        layout.addWidget(self.vtkWidget)

        self.loadButton = QPushButton("Load PLY File", self.centralWidget)
        self.loadButton.clicked.connect(self.load_ply_file)
        layout.addWidget(self.loadButton)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        self.pivot_point = [0, 0, 0]
        self.kd_tree = None  # KDTree for fast nearest neighbor search
        self.points = None  # Store the raw point cloud
        self.update_interactor()
        
    def zoom_camera(self, factor):
        """Zooms while keeping the pivot fixed."""
        camera = self.renderer.GetActiveCamera()  # Ensure we use the correct camera reference
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
        # file_path, _ = QFileDialog.getOpenFileName(self, "Open PLY File", "", "PLY Files (*.ply)")
        file_path = "./data/demo.ply"
        if file_path:
            print(f"Loading: {file_path}")
            ply_data = PlyData.read(file_path)
            self.display_ply(ply_data)

    def display_ply(self, ply_data):
        """Fast PLY Loading with Colors"""
        vertex_data = ply_data['vertex']
        points = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

        # Check if RGB fields exist in the PLY file
        has_color = all(c in vertex_data.data.dtype.names for c in ('red', 'green', 'blue'))
        if has_color:
            colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T 
        else:
            colors = np.full((len(points), 3), [1.0, 1.0, 1.0])  # Default to white

        # Convert NumPy arrays to VTK structures
        vtk_points = vtk.vtkPoints()
        points_vtk = ns.numpy_to_vtk(points, deep=False)
        colors_vtk = ns.numpy_to_vtk(colors.astype(np.uint8), deep=False)

        

        vtk_points.SetData(points_vtk)

        # full-res
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.GetPointData().SetScalars(colors_vtk)

        # Efficient rendering using vtkGlyph3D
        vertex = vtk.vtkVertexGlyphFilter()
        vertex.SetInputData(poly_data)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.Update()

        # low-res mapepr
        points_vtk_lod0 = ns.numpy_to_vtk(points[::100,...], deep=False)
        colors_vtk_lod0 = ns.numpy_to_vtk(colors.astype(np.uint8)[::100,...], deep=False)
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



        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.AddLODMapper(mapper_lod0)
        actor.GetProperty().SetPointSize(2)  # Adjust point size for clarity

        # Clear previous actors and add new
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.set_pivot_point(np.mean(points, axis=0).tolist())  # Set initial pivot
        self.update_interactor()
        self.vtkWidget.GetRenderWindow().Render()





    def get_depth_at_pixel(self, x, y):
        """Fetches the depth buffer value at a given screen pixel using vtk_to_numpy."""
        window = self.vtkWidget.GetRenderWindow()
        window.Render()

        z_buffer = vtk.vtkFloatArray()  # Allocate correct VTK array type

        # Get the depth buffer
        success = window.GetZbufferData(x, x + 1, y, y + 1, z_buffer)
        
        if not success or z_buffer.GetNumberOfTuples() == 0:
            return None  # No valid depth data

        # Convert VTK array to NumPy array
        depth_array = ns.vtk_to_numpy(z_buffer)

        if depth_array.size == 0:
            return None  # No depth data available

        depth = depth_array.mean()  # Use mean if multiple values exist

        print(f"Depth at ({x}, {y}): {depth}")
        
        return None if depth >= 1.0 else depth  # Ignore background (depth = 1.0)

    
    def unproject(self, x, y, depth):
        """Converts 2D screen coordinates to 3D world coordinates using camera matrices."""
        window_size = self.vtkWidget.size()
        aspect_ratio = window_size.width() / window_size.height()

        renderer = self.renderer
        camera = renderer.GetActiveCamera()

        # Convert VTK matrices to NumPy arrays
        view_matrix = np.array(camera.GetModelViewTransformMatrix().GetData()).reshape(4, 4)
        proj_matrix = np.array(camera.GetProjectionTransformMatrix(aspect_ratio, -1, 1).GetData()).reshape(4, 4)
        
        inv_matrix = np.linalg.inv(proj_matrix @ view_matrix)

        # Convert screen coordinates to Normalized Device Coordinates (NDC)
        ndc = np.array([
            (2 * x / window_size.width()) - 1,
            (2 * y / window_size.height()) - 1,
            2 * depth - 1,
            1.0
        ])
        
        world_coords = inv_matrix @ ndc
        return world_coords[:3] / world_coords[3]  # Convert homogeneous to Cartesian coordinates


    def get_nearest_point(self, world_point):
        """Find the nearest point in the point cloud using KDTree."""
        if self.kd_tree is None:
            return None

        distance, index = self.kd_tree.query(world_point)
        return self.points[index]

    def set_pivot_point(self, new_pivot):
        """Updates the pivot point dynamically."""
        self.pivot_point = list(new_pivot)
        self.renderer.GetActiveCamera().SetFocalPoint(*new_pivot)
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def update_interactor(self):
        """Sets the interactor style with the current pivot point."""
        interactor_style = CloudCompareStyleInteractor(self.renderer, self)
        self.interactor.SetInteractorStyle(interactor_style)


if __name__ == "__main__":
    app = QApplication([])
    window = VTKPointCloudViewer()
    window.show()
    app.exec_()


