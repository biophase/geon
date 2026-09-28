"""Optional GPU color blending, isolated from the point-cloud pipeline."""

import numpy as np
import vtk
from vtk.util import numpy_support as ns


class PointCloudBlend:
    ARRAY = "geonBackgroundRGB"
    UNIFORM = "geonForegroundMix"

    def __init__(self, actor: vtk.vtkActor,
                 pipelines: list[tuple[vtk.vtkPolyData, vtk.vtkPolyDataMapper, int]]) -> None:
        self.actor = actor
        self.pipelines = pipelines
        self.enabled = False

    def enable(self, background_rgb: np.ndarray, mix: float) -> None:
        for poly, mapper, stride in self.pipelines:
            colors = np.ascontiguousarray(background_rgb[::stride], dtype=np.float32) / 255.0
            array = ns.numpy_to_vtk(colors, deep=True)
            array.SetName(self.ARRAY)
            poly.GetPointData().AddArray(array)
            mapper.MapDataArrayToVertexAttribute(
                self.ARRAY, self.ARRAY, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1
            )
        shader = self.actor.GetShaderProperty()
        if not self.enabled:
            shader.AddVertexShaderReplacement(
                "//VTK::Color::Dec", True,
                "//VTK::Color::Dec\nin vec3 geonBackgroundRGB;\n", False,
            )
            shader.AddVertexShaderReplacement(
                "//VTK::Color::Impl", True,
                "//VTK::Color::Impl\n"
                "vertexColorVSOutput.rgb = mix(geonBackgroundRGB, "
                "vertexColorVSOutput.rgb, geonForegroundMix);\n", False,
            )
            self.enabled = True
        self.set_mix(mix)

    def set_mix(self, mix: float) -> None:
        if self.enabled:
            self.actor.GetShaderProperty().GetVertexCustomUniforms().SetUniformf(
                self.UNIFORM, mix
            )

    def disable(self) -> None:
        if not self.enabled:
            return
        shader = self.actor.GetShaderProperty()
        shader.ClearVertexShaderReplacement("//VTK::Color::Dec", True)
        shader.ClearVertexShaderReplacement("//VTK::Color::Impl", True)
        shader.GetVertexCustomUniforms().RemoveUniform(self.UNIFORM)
        for poly, mapper, _ in self.pipelines:
            mapper.RemoveVertexAttributeMapping(self.ARRAY)
            poly.GetPointData().RemoveArray(self.ARRAY)
        self.enabled = False
