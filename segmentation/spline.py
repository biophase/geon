import numpy as np
from geomdl import BSpline, utilities

from scipy.spatial import KDTree
import vtk
from typing import Optional

class SplineVolume:
    """
    A spline with frames at its control points.
    Each control point is associated with a frame (an orthogonal rectangle) defined by a width and height.
    The sweeping surface between these frames defines the SplineVolume.
    Its primary purpose is to provide a normalization coordinate system for long, potentially curved infrastructure point clouds.
    """

    def __init__(self, control_points: Optional[np.ndarray] = None,
                 control_handles: Optional[np.ndarray] = None,
                 frame_widths: Optional[np.ndarray] = None,
                 frame_heights: Optional[np.ndarray] = None,
                 global_width: float = 1.0, global_height: float = 1.0,
                 name:str = "Spline Volume"
                 ):
        """
        Initialize the SplineVolume.
        
        Parameters:
          control_points: (N,3) array of control point coordinates.
          control_handles: (optional) additional handle data for each control point.
          frame_widths: Array of length N with per-control point width parameters.
                        Use a negative value (e.g. -1) to indicate default usage.
          frame_heights: Array of length N with per-control point height parameters.
                         Use -1 to denote global default.
          global_width: Global default width.
          global_height: Global default height.
        """
        if control_points is None:
            self.control_points = np.empty((0, 3))
        else:
            self.control_points = control_points
        self.control_handles = control_handles  # for future enhancements

        # Set frame metadata: use provided arrays or default to an array of -1's if not provided.
        n = len(self.control_points)
        if frame_widths is None:
            self._control_points_frame_widths = np.full((n,), -1.0)
        else:
            self._control_points_frame_widths = np.array(frame_widths)
        if frame_heights is None:
            self._control_points_frame_heights = np.full((n,), -1.0)
        else:
            self._control_points_frame_heights = np.array(frame_heights)

        self._frame_width_global = global_width
        self._frame_height_global = global_height

        # Initialize the internal geomdl B-Spline curve.
        self.spline = BSpline.Curve()
        self.spline.degree = 3  # assuming a cubic curve
        if n > 0:
            # geomdl expects a list of control points
            self.spline.ctrlpts = self.control_points.tolist()
            # Generate a uniform (or chord-length based) knot vector automatically.
            self.spline.knotvector = utilities.generate_knot_vector(self.spline.degree, n)
        else:
            self.spline.ctrlpts = []
            self.spline.knotvector = []

        # Placeholders for sampled data
        self.samples_xyz = None       # (num_samples, 3) array of points along the curve
        self.samples_t = None         # corresponding parameter values of samples
        self.samples_frame_x = None   # (num_samples, 3) interpolated frame X vectors
        self.samples_frame_y = None   # (num_samples, 3) interpolated frame Y vectors
        self.samples_kdtree = None    # scipy.spatial.KDTree for fast nearest-neighbor queries

        # for integration in viewer
        self.seg_type = "spline"


    @property
    def control_points_frame_widths(self) -> np.ndarray:
        """
        Return frame widths for control points.
        Every entry of -1 is replaced with the global default width.
        """
        widths = self._control_points_frame_widths.copy()
        widths[widths < 0] = self._frame_width_global
        return widths

    @property
    def control_points_frame_heights(self) -> np.ndarray:
        """
        Return frame heights for control points.
        Every entry of -1 is replaced with the global default height.
        """
        heights = self._control_points_frame_heights.copy()
        heights[heights < 0] = self._frame_height_global
        return heights

    def set_control_frame_X(self, control_frame_i: Optional[int], new_width: float) -> None:
        """
        Set the width for a control frame.
        If control_frame_i is None, adjust the global default.
        """
        if control_frame_i is None:
            self._frame_width_global = new_width
        else:
            if control_frame_i < 0 or control_frame_i >= len(self._control_points_frame_widths):
                raise IndexError("Control frame index out of range")
            self._control_points_frame_widths[control_frame_i] = new_width

    def set_control_frame_Y(self, control_frame_i: Optional[int], new_height: float) -> None:
        """
        Set the height for a control frame.
        If control_frame_i is None, adjust the global default.
        """
        if control_frame_i is None:
            self._frame_height_global = new_height
        else:
            if control_frame_i < 0 or control_frame_i >= len(self._control_points_frame_heights):
                raise IndexError("Control frame index out of range")
            self._control_points_frame_heights[control_frame_i] = new_height

    @classmethod
    def draw_xy(cls, z: float):
        """
        (Class method) Initiate drawing a spline in the XY plane at a given Z height.
        In an interactive application, this would trigger a drawing mode.
        Here we simply create an instance and store a default z.
        """
        instance = cls()
        instance.default_z = z  # store z-level for drawing purposes
        return instance

    def sample_spline(self, num_samples: Optional[int] = None, sample_spacing: Optional[float] = None) -> None:
        """
        Sample the spline at regular intervals.
        
        Parameters:
          num_samples: The number of samples to generate along the curve.
          sample_spacing: Alternatively, provide a spacing (in units of curve length) to determine sample count.
          
        Exactly one of these must be provided (if both are provided, num_samples is used).
        """
        if num_samples is None and sample_spacing is None:
            raise ValueError("Either num_samples or sample_spacing must be provided")

        if num_samples is not None:
            parameter_values = np.linspace(0, 1, num_samples)
        else:
            # Estimate the total length of the spline for spacing calculation.
            dense_t = np.linspace(0, 1, 1000)
            dense_pts = np.array(self.spline.evaluate_list(dense_t))
            dists = np.sqrt(np.sum(np.diff(dense_pts, axis=0)**2, axis=1))
            total_length = np.sum(dists)
            num_samples = int(total_length / sample_spacing) + 1
            parameter_values = np.linspace(0, 1, num_samples)

        # Evaluate the spline points
        self.samples_xyz = np.array(self.spline.evaluate_list(parameter_values))
        self.samples_t = parameter_values

        # Compute derivatives (tangent vectors) at each sampled parameter.
        derivatives = np.array([self.spline.derivatives(t, order=1)[1] for t in parameter_values])
        frame_x_list = []
        frame_y_list = []

        # Interpolate the per-control-point frame widths and heights along the parameter
        cp_count = len(self.control_points)
        param_cp = np.linspace(0, 1, cp_count) if cp_count > 1 else np.array([0, 1])
        widths = np.interp(parameter_values, param_cp, self.control_points_frame_widths)
        heights = np.interp(parameter_values, param_cp, self.control_points_frame_heights)

        # For each sample, compute the local frame.
        for i, tangent in enumerate(derivatives):
            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                tangent_norm = np.array([1, 0, 0])
            else:
                tangent_norm = tangent / norm

            # We want the frame's X vector to be horizontal.
            # Project the tangent onto the XY plane and create a perpendicular vector.
            tangent_xy = np.array([tangent_norm[0], tangent_norm[1], 0])
            if np.linalg.norm(tangent_xy) < 1e-6:
                perp = np.array([1, 0, 0])
            else:
                tangent_xy_norm = tangent_xy / np.linalg.norm(tangent_xy)
                perp = np.array([-tangent_xy_norm[1], tangent_xy_norm[0], 0])
            frame_x = perp * widths[i]

            # Frame Y is computed using the cross-product to ensure orthogonality.
            # (You may refine this if a different definition is required.)
            frame_y = np.cross(tangent_norm, perp)
            frame_y = frame_y * heights[i]

            frame_x_list.append(frame_x)
            frame_y_list.append(frame_y)

        self.samples_frame_x = np.array(frame_x_list)
        self.samples_frame_y = np.array(frame_y_list)

        # Build the KDTree from the sampled points for fast nearest-neighbor queries.
        self.samples_kdtree = KDTree(self.samples_xyz)

    def visualize_vtk(self, renderer, **kwargs):
        """
        Create a VTK visualization of the spline.
        
        This example builds a simple vtkPolyData showing the spline as a red polyline.
        It returns the created vtkActor so that further interactive editing or synchronization
        with a custom "vtkSpline" widget can be achieved.
        
        Parameters:
          renderer: A vtkRenderer instance in which to add the actor.
          kwargs: Additional keyword arguments for further customization.
        """
        if self.samples_xyz is None:
            raise ValueError("Spline must be sampled first. Call sample_spline() before visualization.")

        # Create vtkPoints from the sampled points
        points = vtk.vtkPoints()
        for pt in self.samples_xyz:
            points.InsertNextPoint(pt[0], pt[1], pt[2])

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)

        # Create a polyline that connects the sample points.
        polyLine = vtk.vtkPolyLine()
        n = self.samples_xyz.shape[0]
        polyLine.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyLine.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)
        polyData.SetLines(cells)

        # Mapper and actor for VTK visualization.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # set spline color to red
        actor.GetProperty().SetLineWidth(2)

        renderer.AddActor(actor)
        return actor

    def compute_normalized_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute normalized coordinates for an input point cloud using the spline as a reference.
        
        For each incoming point (N,3):
         1. Find the nearest sampled spline point.
         2. Use that sample's local frame (defined by the X and Y vectors) to project the
            point onto the plane.
         3. The normalized (x, y) are computed as the dot-product ratios (i.e. how many frame lengths away from the center).
         4. The third coordinate is the parameter (t value) of the closest sampled point.
        
        The operation is vectorized (using KDTree queries, broadcasting, etc.) and does not use explicit loops.
        
        Parameters:
          coords: (N,3) numpy array of incoming points.
          
        Returns:
          normalized_coords: (N,3) array where the first two columns are normalized in the frame space
                             (with “1” meaning an exact reach of the frame edge) and the third column is the t value.
        """
        if self.samples_kdtree is None or self.samples_xyz is None:
            raise ValueError("Spline has not been sampled; please run sample_spline() first.")

        # Find for each point the index of the closest sample along the spline.
        distances, indices = self.samples_kdtree.query(coords)
        closest_pts = self.samples_xyz[indices]
        frame_x = self.samples_frame_x[indices]
        frame_y = self.samples_frame_y[indices]
        t_values = self.samples_t[indices]

        # Compute the difference vectors
        vectors = coords - closest_pts

        # Compute projections onto the local frame components.
        # We use the dot product divided by the squared norm of the frame vector so that a point exactly at the edge gives 1.
        norm_x = np.linalg.norm(frame_x, axis=1)
        norm_y = np.linalg.norm(frame_y, axis=1)
        norm_x[norm_x == 0] = 1  # safeguard against division by zero
        norm_y[norm_y == 0] = 1

        proj_x = np.einsum('ij,ij->i', vectors, frame_x) / (norm_x**2)
        proj_y = np.einsum('ij,ij->i', vectors, frame_y) / (norm_y**2)

        # Combine the projections and the t parameter into a normalized coordinate.
        normalized_coords = np.column_stack((proj_x, proj_y, t_values))
        return normalized_coords
