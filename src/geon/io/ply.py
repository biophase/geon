from __future__ import annotations
from typing import Dict, Tuple, Mapping, Sequence



import numpy as np
from plyfile import PlyData, PlyElement

from geon.data.pointcloud import PointCloudData, FieldType


PLY_DTYPE_CHOICES: dict[str, np.dtype] = {
    "int8": np.dtype("i1"),
    "uint8": np.dtype("u1"),
    "int16": np.dtype("i2"),
    "uint16": np.dtype("u2"),
    "int32": np.dtype("i4"),
    "uint32": np.dtype("u4"),
    "float32": np.dtype("f4"),
    "float64": np.dtype("f8"),
}


def _safe_property_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip())
    return out or "field"


def _component_names(field_name: str, component_count: int, field_type: FieldType) -> list[str]:
    base = _safe_property_name(field_name)
    if component_count == 1:
        return [base]
    if field_type == FieldType.COLOR and component_count >= 3:
        names = ["red", "green", "blue"]
        names.extend(f"{base}_{i}" for i in range(3, component_count))
        return names[:component_count]
    if field_type == FieldType.NORMAL and component_count == 3:
        return ["nx", "ny", "nz"]
    return [f"{base}_{i}" for i in range(component_count)]


def pcd_to_ply(
    pcd: PointCloudData,
    path: str,
    *,
    field_names: Sequence[str] | None = None,
    field_dtypes: Mapping[str, str | np.dtype] | None = None,
    coord_dtype: str | np.dtype | None = None,
    text: bool = False,
) -> None:
    """
    Save a PointCloudData object as a PLY vertex element.

    Multi-component GEON fields are exported as multiple scalar PLY
    properties using stable component suffixes. Color fields use
    red/green/blue and normal fields use nx/ny/nz where possible.
    """
    num_points = pcd.points.shape[0]
    if pcd.points.ndim != 2 or pcd.points.shape[1] < 3:
        raise ValueError("PointCloudData.points must have shape (N, >=3).")

    selected = set(field_names) if field_names is not None else set(pcd.field_names)
    dtype_map = dict(field_dtypes or {})
    coord_np_dtype = np.dtype(coord_dtype) if coord_dtype is not None else pcd.points.dtype

    dtype_fields: list[tuple[str, np.dtype]] = []
    assignments: list[tuple[str, np.ndarray]] = []
    used_names: set[str] = set()

    def add_property(name: str, values: np.ndarray, dtype: np.dtype) -> None:
        prop_name = _safe_property_name(name)
        if prop_name in used_names:
            suffix = 1
            while f"{prop_name}_{suffix}" in used_names:
                suffix += 1
            prop_name = f"{prop_name}_{suffix}"
        used_names.add(prop_name)
        dtype_fields.append((prop_name, dtype))
        assignments.append((prop_name, values.astype(dtype, copy=False)))

    for idx, coord_name in enumerate(("x", "y", "z")):
        add_property(coord_name, pcd.points[:, idx], coord_np_dtype)

    for field in pcd.get_fields():
        if field.name not in selected:
            continue
        data = np.asarray(field.data)
        if data.shape[0] != num_points:
            raise ValueError(
                f"Field '{field.name}' has {data.shape[0]} entries, expected {num_points}."
            )
        if data.ndim == 1:
            matrix = data.reshape(num_points, 1)
        elif data.ndim == 2:
            matrix = data
        else:
            raise ValueError(f"Unsupported field dimensionality for '{field.name}': {data.shape}.")

        target_dtype = np.dtype(dtype_map.get(field.name, matrix.dtype))
        for comp_idx, prop_name in enumerate(
            _component_names(field.name, matrix.shape[1], field.field_type)
        ):
            add_property(prop_name, matrix[:, comp_idx], target_dtype)

    vertex_data = np.empty(num_points, dtype=dtype_fields)
    for name, values in assignments:
        vertex_data[name] = values

    vertex = PlyElement.describe(vertex_data, "vertex")
    PlyData([vertex], text=text).write(path)


def ply_to_pcd(
    path: str,
    fields_map: Mapping[str, Tuple[str, int]],
    field_types: Mapping[str, FieldType],
) -> PointCloudData:
    """
    Load a PLY file and convert it into a PointCloudData instance.

    Parameters
    ----------
    path:
        Path to the .ply file.
    fields_map:
        Mapping from *ply field name* -> (internal_field_name, component_index).

        Example:
            {
                "red":   ("Color", 0),
                "green": ("Color", 1),
                "blue":  ("Color", 2),
                "intensity": ("Intensity", 0),
            }

        If multiple PLY fields map to the same internal_field_name with
        different component_index values, they will be combined into a
        multi-dimensional field (shape [N, K]).
        If only one component_index is used, you still get a 2D array
        of shape [N, 1] (as expected by PointCloudData.add_field).

    field_types:
        Mapping from *internal_field_name* -> FieldType.

        Example:
            {
                "Color": FieldType.COLOR,
                "Intensity": FieldType.INTENSITY,
            }

        If a given internal_field_name is not present in field_types,
        the type is inferred:
            - 1 component  -> FieldType.SCALAR
            - >1 component -> FieldType.VECTOR

    Returns
    -------
    PointCloudData
        A new point cloud with points and fields populated.
    """
    ply = PlyData.read(path)

    if "vertex" not in ply:
        raise ValueError(f"PLY file '{path}' has no 'vertex' element.")

    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()

    required_coords = ("x", "y", "z")
    missing_coords = [c for c in required_coords if c not in names]
    if missing_coords:
        raise ValueError(
            f"PLY file '{path}' is missing coordinate fields: {missing_coords}"
        )

    # --- Points (N,3) ---
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    points = np.stack([x, y, z], axis=1)

    pcd = PointCloudData(points)

    # --- Group PLY fields by internal field name ---
    # grouped["Color"] = [(0, arr_red), (1, arr_green), (2, arr_blue)], etc.
    grouped: Dict[str, list[Tuple[int, np.ndarray]]] = {}

    for ply_name, (internal_name, pos) in fields_map.items():
        if ply_name not in names:
            # You can choose to raise here instead if you want strict behavior.
            # For now: silently skip missing PLY fields.
            continue

        arr = np.asarray(vertex[ply_name])
        if arr.ndim != 1:
            # PLY vertex properties are normally 1D (N,). Be defensive.
            arr = arr.reshape(-1)

        grouped.setdefault(internal_name, []).append((pos, arr))

    # --- Build fields for each internal name ---
    for internal_name, components in grouped.items():
        if not components:
            continue

        # Sort by component index (0,1,2,...)
        components_sorted = sorted(components, key=lambda t: t[0])

        # Determine number of components (vector dimension)
        max_pos = components_sorted[-1][0]
        vector_dim = max_pos + 1

        num_points = pcd.points.shape[0]

        # Decide dtype by promoting all component dtypes together
        result_dtype = np.result_type(*[c_arr.dtype for _, c_arr in components_sorted])
        data = np.zeros((num_points, vector_dim), dtype=result_dtype)

        for pos, arr in components_sorted:
            if arr.shape[0] != num_points:
                raise ValueError(
                    f"Field '{internal_name}' component has {arr.shape[0]} entries, "
                    f"but point cloud has {num_points} points."
                )
            data[:, pos] = arr.astype(result_dtype, copy=False)

        # Determine field type: from field_types or infer
        ftype = field_types.get(internal_name)
        if ftype is None:
            ftype = FieldType.SCALAR if vector_dim == 1 else FieldType.VECTOR

        # Add to PointCloudData
        pcd.add_field(
            name=internal_name,
            data=data,
            field_type=ftype,
            vector_dim_hint=vector_dim,
        )

    return pcd
