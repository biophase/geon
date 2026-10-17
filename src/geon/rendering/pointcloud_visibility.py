"""Prototype visibility attribute shared by display and hardware picking."""

import numpy as np
import vtk
from vtk.util import numpy_support as ns


class PointCloudVisibility:
    ARRAY = "geonVisible"
    UNIFORM = "geonVisibilityEnabled"

    def __init__(self, actor, pipelines):
        self.actor = actor
        self.pipelines = pipelines
        shader = actor.GetShaderProperty()
        shader.AddVertexShaderReplacement(
            "//VTK::PositionVC::Dec", True,
            "//VTK::PositionVC::Dec\nin float geonVisible;\n", False)
        shader.AddVertexShaderReplacement(
            "//VTK::PositionVC::Impl", True,
            "//VTK::PositionVC::Impl\n"
            "if (geonVisibilityEnabled != 0 && geonVisible < 0.5) {\n"
            "  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n"
            "}\n", False)
        for poly, mapper, _ in pipelines:
            array = ns.numpy_to_vtk(np.ones(poly.GetNumberOfPoints(), dtype=np.uint8))
            array.SetName(self.ARRAY)
            poly.GetPointData().AddArray(array)
            mapper.MapDataArrayToVertexAttribute(
                self.ARRAY, self.ARRAY, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
        self.set_mask(None)

    def set_mask(self, mask):
        # Show all needs only a uniform update, leaving geometry and buffers intact.
        for poly, _, stride in self.pipelines:
            current = poly.GetPointData().GetArray(self.ARRAY)
            if mask is not None or current.GetNumberOfTuples() != poly.GetNumberOfPoints():
                values = (np.ones(poly.GetNumberOfPoints(), dtype=np.uint8) if mask is None
                          else np.ascontiguousarray(mask[::stride], dtype=np.uint8))
                array = ns.numpy_to_vtk(values, deep=False)
                array.SetName(self.ARRAY)
                poly.GetPointData().AddArray(array)
        self.actor.GetShaderProperty().GetVertexCustomUniforms().SetUniformi(
            self.UNIFORM, int(mask is not None))
