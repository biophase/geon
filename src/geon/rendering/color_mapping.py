"""Shared map resolution and small GPU lookup tables for point fields."""
from copy import deepcopy
import numpy as np
import vtk
from vtk.util import numpy_support as ns
from geon.data.definitions import (ColorMap, VectorColorMap, vector_preset,
                                   finite_extrema, resolve_limits, sample_map)
from geon.data.pointcloud import FieldType


def default_map(field):
    if field.color_map is not None:
        return field.color_map
    if field.field_type == FieldType.NORMAL:
        return vector_preset(-1, 1)
    if field.field_type == FieldType.COLOR:
        low, high = finite_extrema(field.data)
        return vector_preset(0, 1 if low >= 0 and high <= 1 else 255)
    return ColorMap.get_cmap('gray' if field.field_type == FieldType.INTENSITY else 'rainbow')


def components(field, mapping):
    count = 1 if field.data.ndim == 1 else field.data.shape[1]
    indices = field.rgb_components if isinstance(mapping, VectorColorMap) else (field.scalar_component,)
    if any(i < 0 or i >= count for i in indices):
        raise ValueError('Input component does not exist in this field')
    data = field.data.reshape(len(field.data), count)
    return [data[:, i] for i in indices]


def linear_endpoints(mapping):
    """Return exact affine outputs, or None for curves requiring the lookup path.

    Deliberately use exact comparisons: a nearly linear edited curve is still a
    curve. Redundant collinear stops do not prevent the fast path.
    """
    mapping.validate()
    if isinstance(mapping, VectorColorMap):
        curves = [(m.color_positions, m.output_values) for m in mapping.channels]
    elif mapping.color_type == 'rgb' and np.array_equal(
            mapping.color_definitions, np.repeat(mapping.color_definitions[:, :1], 3, axis=1)):
        curves = [(mapping.color_positions, mapping.color_definitions[:, 0])] * 3
    else:
        return None
    for positions, values in curves:
        expected = values[0] + positions * (values[-1] - values[0])
        if len(positions) > 2 and not np.array_equal(values, expected):
            return None
    return (tuple(float(values[0]) for _, values in curves),
            tuple(float(values[-1]) for _, values in curves))


class GPUColorMapping:
    """Value attributes stay resident while map uniforms change during editing."""
    SIZE = 128

    def __init__(self, layer):
        self.layer = layer
        self.states = {}
        shader = layer._main_actor.GetShaderProperty()
        declarations = ''
        implementation = ''
        for slot in ('Foreground', 'Background'):
            prefix = 'geon' + slot
            declarations += f'\nin vec3 {prefix}Input;\n'
            declarations += f'''
vec3 {prefix}Evaluate(vec3 v) {{
  if ({prefix}Mode == 0) return v;
  if (any(isnan(v)) || any(isinf(v))) return vec3(0.6);
  // Linear grayscale only needs the scalar input, with no table indexing.
  if ({prefix}Mode == 3) {{
    float span = {prefix}Max.x - {prefix}Min.x;
    float t = span == 0.0 ? 0.5 : clamp((v.x-{prefix}Min.x)/span,0.0,1.0);
    return mix({prefix}Start, {prefix}End, t);
  }}
  vec3 t;
  for (int c=0; c<3; ++c) {{
    float span = {prefix}Max[c] - {prefix}Min[c];
    t[c] = span == 0.0 ? 0.5 : clamp((v[c]-{prefix}Min[c])/span,0.0,1.0);
  }}
  if ({prefix}Mode == 1) t = vec3(t.x);
  if ({prefix}Mode == 4) return mix({prefix}Start, {prefix}End, t);
  vec3 color;
  for (int c=0; c<3; ++c) {{
    float x = t[c] * {self.SIZE-1}.0;
    int a = int(floor(x)); int b = min(a+1,{self.SIZE-1});
    color[c] = mix({prefix}Table[a*3+c], {prefix}Table[b*3+c], fract(x));
  }}
  return color;
}}
'''
        implementation = '''
vec3 geonMappedColor = geonForegroundEvaluate(geonForegroundInput);
if (geonMapMix < 1.0) {
  geonMappedColor = mix(geonBackgroundEvaluate(geonBackgroundInput), geonMappedColor, geonMapMix);
}
vertexColorVSOutput = vec4(geonMappedColor, 1.0);
'''
        self.declarations, self.implementation = declarations, implementation
        shader.AddVertexShaderReplacement('//VTK::Color::Dec', True,
                                         '//VTK::Color::Dec\n' + declarations, False)
        shader.AddVertexShaderReplacement('//VTK::Color::Impl', True,
                                         '//VTK::Color::Impl\n' + implementation, False)
        self.uniforms = shader.GetVertexCustomUniforms()
        for slot in ('Foreground', 'Background'):
            prefix = 'geon' + slot
            self.uniforms.SetUniformi(prefix+'Mode', 0)
            self.uniforms.SetUniform3f(prefix+'Min', (0.,)*3)
            self.uniforms.SetUniform3f(prefix+'Max', (1.,)*3)
            self.uniforms.SetUniform3f(prefix+'Start', (0.,)*3)
            self.uniforms.SetUniform3f(prefix+'End', (1.,)*3)
            self.uniforms.SetUniform1fv(prefix+'Table', self.SIZE*3, [0.]*(self.SIZE*3))
        self.uniforms.SetUniformf('geonMapMix', 1.)

    def set_mix(self, amount):
        self.uniforms.SetUniformf('geonMapMix', amount if self.layer.background_field_name else 1.)

    def upload(self, slot, field):
        layer = self.layer
        if slot == 'Background' and field is None:
            self.states.pop(slot, None)
            self.uniforms.SetUniformi('geonBackgroundMode', 0)
            for poly, mapper, _ in layer._blend.pipelines:
                source = poly.GetPointData().GetArray('geonForegroundInput')
                array = source.NewInstance()
                array.ShallowCopy(source)
                array.SetName('geonBackgroundInput')
                poly.GetPointData().AddArray(array)
                mapper.MapDataArrayToVertexAttribute('geonBackgroundInput', 'geonBackgroundInput', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
            return
        if field is not None:
            field = layer._color_overrides.get(field.name, field)
        mapped = field is not None and field.field_type not in (FieldType.SEMANTIC, FieldType.INSTANCE)
        if mapped:
            mapping = default_map(field)
            values = components(field, mapping)
            extrema = [finite_extrema(v) for v in values]
            inputs = (np.column_stack(values).astype(np.float32) if len(values) == 3
                      else np.ascontiguousarray(values[0], dtype=np.float32))
            self.states[slot] = (field, extrema)
        else:
            display = layer._field_display(field)
            inputs = layer._display_rgb(display, len(layer.data.points)).astype(np.float32) / 255.
            self.states.pop(slot, None)
            self.uniforms.SetUniformi('geon'+slot+'Mode', 0)
        name = 'geon'+slot+'Input'
        for poly, mapper, stride in layer._blend.pipelines:
            array = ns.numpy_to_vtk(np.ascontiguousarray(inputs[::stride]), deep=False)
            array.SetName(name)
            if slot == 'Foreground':
                # VTK needs a color scalar to declare its color varying.
                scalar = array
                if not mapped:
                    scalar = ns.numpy_to_vtk(np.ascontiguousarray(np.clip(inputs[::stride]*255,0,255), dtype=np.uint8))
                poly.GetPointData().SetScalars(scalar)
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
                mapper.SetLookupTable(None)
            poly.GetPointData().AddArray(array)
            mapper.MapDataArrayToVertexAttribute(name, name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
        if mapped:
            self.update_map(slot, mapping)

    def update_map(self, slot, mapping):
        field, extrema = self.states[slot]
        prefix = 'geon'+slot
        vector = isinstance(mapping, VectorColorMap)
        maps = mapping.channels if vector else (mapping,)
        limits = [resolve_limits(m, e) for m, e in zip(maps, extrema)]
        if not vector:
            limits *= 3
        self.uniforms.SetUniform3f(prefix+'Min', tuple(v[0] for v in limits))
        self.uniforms.SetUniform3f(prefix+'Max', tuple(v[1] for v in limits))
        endpoints = linear_endpoints(mapping)
        if endpoints is not None:
            self.uniforms.SetUniform3f(prefix+'Start', endpoints[0])
            self.uniforms.SetUniform3f(prefix+'End', endpoints[1])
            self.uniforms.SetUniformi(prefix+'Mode', 4 if vector else 3)
            return
        t = np.linspace(0, 1, self.SIZE)
        table = np.column_stack([sample_map(m, t) for m in maps]) if vector else sample_map(mapping, t)
        self.uniforms.SetUniformi(prefix+'Mode', 2 if vector else 1)
        self.uniforms.SetUniform1fv(prefix+'Table', table.size, table.ravel().tolist())

    def refresh(self):
        shader = self.layer._main_actor.GetShaderProperty()
        shader.AddVertexShaderReplacement('//VTK::Color::Dec', True,
            '//VTK::Color::Dec\n'+self.declarations, False)
        shader.AddVertexShaderReplacement('//VTK::Color::Impl', True,
            '//VTK::Color::Impl\n'+self.implementation, False)
        self.upload('Foreground', self.layer.active_field)
        name = self.layer.background_field_name
        field = self.layer.data.get_fields(name)[0] if name else None
        self.upload('Background', field)
        self.set_mix(self.layer.foreground_mix)

    def preview(self, field, mapping):
        for slot, (candidate, _) in self.states.items():
            if candidate.name == field.name:
                self.update_map(slot, mapping)
        self.layer.signals.colorDisplayChanged.emit()
