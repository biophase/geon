"""Selection highlighting on the existing cloud, without overlay geometry."""

import math
import numpy as np
import vtk
from PyQt6.QtCore import QTimer
from vtk.util import numpy_support as ns

from geon.util.operation_timing import timed  # TEMP_VISIBILITY_PROFILE


class PointCloudSelection:
    ARRAY = "geonSelected"
    ENABLED = "geonSelectionEnabled"
    PULSE = "geonSelectionPulse"

    def __init__(self, layer):
        self.layer = layer
        self.selection_timer = QTimer()
        self.selection_timer.setInterval(50)
        self.selection_timer.timeout.connect(self.update_pulse)
        self.phase = 0.0

    def attach(self):
        shader = self.layer._main_actor.GetShaderProperty()
        # Use separate replacement hooks from color blending and visibility.
        shader.AddVertexShaderReplacement(
            "//VTK::TCoord::Dec", True,
            "//VTK::TCoord::Dec\nin float geonSelected;\n"
            "out float geonSelectedVSOutput;\n", False)
        shader.AddVertexShaderReplacement(
            "//VTK::TCoord::Impl", True,
            "//VTK::TCoord::Impl\ngeonSelectedVSOutput = geonSelected;\n", False)
        shader.AddFragmentShaderReplacement(
            "//VTK::TCoord::Dec", True,
            "//VTK::TCoord::Dec\nin float geonSelectedVSOutput;\n", False)
        shader.AddFragmentShaderReplacement(
            "//VTK::Light::Impl", True,
            "//VTK::Light::Impl\n"
            "if (geonSelectionEnabled != 0 && geonSelectedVSOutput > 0.5) {\n"
            "  gl_FragData[0].rgb = mix(gl_FragData[0].rgb, vec3(1.0, 0.5, 0.0), geonSelectionPulse);\n"
            "}\n", False)
        self.set_pulse(1.0)
        self.update_mask()

    def set_pulse(self, amount):
        actor = self.layer._main_actor
        if actor is not None:
            actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
                self.PULSE, float(amount))

    @timed('Update GPU selection mask')  # TEMP_VISIBILITY_PROFILE
    def update_mask(self):
        layer = self.layer
        if layer._main_actor is None:
            return
        selection = layer.active_selection
        enabled = selection is not None and selection.size > 0
        mask = None
        if enabled:
            mask = np.zeros(len(layer.data.points), dtype=np.uint8)
            mask[selection] = 1
        for poly, mapper, stride in layer._blend.pipelines:
            current = poly.GetPointData().GetArray(self.ARRAY)
            # Deselect uses only a uniform; allocate zero attributes only on setup/resize.
            if enabled or current is None or current.GetNumberOfTuples() != poly.GetNumberOfPoints():
                values = (np.ascontiguousarray(mask[::stride]) if enabled else
                          np.zeros(poly.GetNumberOfPoints(), dtype=np.uint8))
                array = ns.numpy_to_vtk(values, deep=False)
                array.SetName(self.ARRAY)
                poly.GetPointData().AddArray(array)
            if current is None:
                mapper.MapDataArrayToVertexAttribute(
                    self.ARRAY, self.ARRAY, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
        layer._main_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformi(
            self.ENABLED, int(enabled))
        if enabled and layer.renderer is not None:
            if not self.selection_timer.isActive():
                self.phase = 0.0
                self.set_pulse(1.0)
                self.selection_timer.start()
        else:
            self.selection_timer.stop()

    def update_pulse(self):
        layer = self.layer
        if layer.renderer is None or layer.active_selection is None or not layer.active_selection.size:
            self.selection_timer.stop()
            return
        self.phase += 0.1
        self.set_pulse((math.cos(self.phase * 4) + 1) / 2)
        window = layer.renderer.GetRenderWindow()
        if window is not None and layer.visible:
            window.Render()

    def detach(self):
        self.selection_timer.stop()
