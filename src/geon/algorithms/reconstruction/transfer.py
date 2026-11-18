"""Transfer fields to final faces and build disposable surface deviation samples."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.strtree import STRtree

from geon.data.mesh import MeshData, MeshLevel
from geon.data.pointcloud import FieldType
from geon.util.resources import resource_path
from .core import Config, reconstruct
from .primitives import curve_in_patch


def default_config():
    with open(resource_path("reconstruction_settings.json"), encoding="utf-8") as stream:
        return Config(**json.load(stream))


def _groups(values):
    order = np.argsort(values, kind="stable")
    ids, starts = np.unique(values[order], return_index=True)
    return {int(label): group for label, group in zip(ids, np.split(order, starts[1:]))}


def assign_points_to_faces(result, points, primitive_ids, mask=None, progress=None):
    """Project source points into each primitive's final triangles.

    STRtree queries are vectorized in bounded chunks. Points outside a simplified
    footprint vote for the closest final face in the same primitive. This avoids
    assigning across adjoining surfaces and avoids centroid-distance heuristics.
    """
    points = np.asarray(points)
    assignment = np.full(len(points), -1, dtype=np.int64)
    valid = np.isfinite(points).all(axis=1)
    if mask is not None:
        valid &= mask
    source = np.flatnonzero(valid)
    source_groups = _groups(np.asarray(primitive_ids)[source])
    face_groups = _groups(result.face_patch_ids)
    for label, face_indices in face_groups.items():
        if progress:
            progress(f"Transferring source points to final faces: primitive {label}")
        if label not in source_groups or label not in result.patches:
            continue
        point_indices = source[source_groups[label]]
        patch = result.patches[label]
        triangles = result.vertices[result.faces[face_indices]]
        uv_triangles = np.array([curve_in_patch(patch, tri) for tri in triangles])
        polygons = shapely.polygons(uv_triangles)
        tree = STRtree(polygons)
        for start in range(0, len(point_indices), 100_000):
            rows = point_indices[start:start+100_000]
            uv = patch.project(points[rows])
            if patch.cylinder is not None and patch.cylinder.periodic:
                period = 2*np.pi*patch.cylinder.radius
                center = np.mean(uv_triangles[:, :, 0])
                uv[:, 0] += np.round((center-uv[:, 0])/period)*period
            geometry = shapely.points(uv)
            pairs = tree.query(geometry, predicate="intersects")
            local = np.full(len(rows), len(face_indices), dtype=np.int64)
            if pairs.size:
                np.minimum.at(local, pairs[0], pairs[1])
            missing = local == len(face_indices)
            if missing.any():
                local[missing] = tree.nearest(geometry[missing])
            assignment[rows] = face_indices[local]
    return assignment


def reconstruct_cloud(cloud, primitive_name, mapped_names=(), *, config=None, mask=None, progress=None):
    fields = cloud.get_fields(names=primitive_name, field_type=FieldType.INSTANCE)
    if len(fields) != 1:
        raise ValueError("Choose an instance field for primitive segmentation.")
    config = config or default_config()
    primitive_ids = fields[0].data.reshape(-1)
    result = reconstruct(cloud.points, primitive_ids, mask=mask, config=config, progress=progress)
    if not len(result.faces):
        reasons = '\n'.join(message for message in result.diagnostics
                            if message.startswith('Patch ') or message.startswith('Size filter:'))
        raise ValueError("Reconstruction produced no faces.\n" +
                         (reasons or "Check the source instance field, selection, and reconstruction settings."))
    mesh = MeshData(result.vertices, result.faces, "Reconstructed mesh",
                    face_object_ids=result.face_patch_ids)
    mesh.metadata = {"reconstruction_settings": asdict(config), "primitive_field": primitive_name,
                     "diagnostics": result.diagnostics}
    primitive = mesh.add_field(primitive_name, result.face_patch_ids, FieldType.INSTANCE)
    primitive.color_map = deepcopy(fields[0].color_map)
    mapped_names = list(dict.fromkeys(name for name in mapped_names if name and name != primitive_name))
    assignment = assign_points_to_faces(result, cloud.points, primitive_ids, mask, progress)
    assigned = np.flatnonzero(assignment >= 0)
    result.deviation_point_indices = assigned
    result.point_deviations = np.empty(len(assigned), dtype=np.float32)
    for start in range(0, len(assigned), 100_000):
        if progress:
            progress("Measuring source-point distance to reconstructed faces")
        rows = assigned[start:start+100_000]
        result.point_deviations[start:start+len(rows)] = point_triangle_distances(
            cloud.points[rows], result.vertices[result.faces[assignment[rows]]])
    groups = _groups(assignment[assigned])
    source_groups = _groups(primitive_ids)
    fallback = {}
    for label, face_indices in _groups(result.face_patch_ids).items():
        indices = source_groups[label]
        valid = np.isfinite(cloud.points[indices]).all(axis=1)
        if mask is not None:
            valid &= mask[indices]
        indices = indices[valid]
        centers = result.vertices[result.faces[face_indices]].mean(axis=1)
        nearest = cKDTree(cloud.points[indices]).query(centers)[1]
        fallback.update(zip(face_indices, indices[nearest]))
    for name in mapped_names:
        if progress:
            progress(f"Voting on final faces: {name}")
        matches = cloud.get_fields(names=name)
        if len(matches) != 1:
            raise ValueError(f"Unknown source field: {name}")
        source_field = matches[0]
        data = source_field.data.reshape(len(cloud.points), -1)
        output = []
        categorical = source_field.field_type in (FieldType.SEMANTIC, FieldType.INSTANCE)
        for face_index in range(len(result.faces)):
            voters = assigned[groups[face_index]] if face_index in groups else [fallback[face_index]]
            values = data[voters]
            if categorical:
                unique, counts = np.unique(values, axis=0, return_counts=True)
                output.append(unique[np.argmax(counts)])  # Ties select the lowest ID.
            else:
                output.append(np.mean(values, axis=0))
        field = deepcopy(source_field)
        field.data = np.asarray(output, dtype=source_field.data.dtype).reshape(len(result.faces), -1)
        mesh.field_container(MeshLevel.FACE)._fields.append(field)
    return mesh, result


def deviation_surface(result, points, primitive_ids, *, spacing=.05, max_vertices=None, mask=None, progress=None):
    """Regular, temporary triangulation colored by local source-to-face RMS.

    Geometry is sampled at the requested spacing without a default size cap. Only the preview owns
    these arrays; the accepted mesh and its HDF5 representation stay compact.
    Neighbour searches are restricted to each source primitive.
    """
    if spacing <= 0 or (max_vertices is not None and max_vertices < 3):
        raise ValueError("Deviation spacing and budget must be positive.")
    triangles = result.vertices[result.faces]
    lengths = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2).max(axis=1)
    divisions = np.maximum(1, np.ceil(lengths/spacing).astype(int))
    if max_vertices is not None and len(triangles)*3 > max_vertices:
        # A budget smaller than the original surface cannot sample every face.
        raise ValueError("Deviation preview budget is smaller than the reconstructed mesh.")
    while max_vertices is not None and np.sum((divisions+1)*(divisions+2)//2) > max_vertices:
        divisions = np.maximum(1, (divisions*.85).astype(int))
    original = result.deviation_point_indices
    groups = _groups(np.asarray(primitive_ids)[original])
    vertices, faces, deviation = [], [], []
    offset = 0
    trees = {}
    for face_index, (triangle, label, n) in enumerate(zip(triangles, result.face_patch_ids, divisions)):
        if progress is not None and face_index % 256 == 0:
            progress(f'Deviation preview: face {face_index+1}/{len(triangles)}')
        label, n = int(label), int(n)
        if label not in trees:
            source = points[original[groups[label]]]
            trees[label] = cKDTree(source), result.point_deviations[groups[label]]
        tree, residuals = trees[label]
        i,k=np.triu_indices(n+1)
        j=k-i
        grid=(triangle[0]+(triangle[1]-triangle[0])*(i/n)[:,None]
              +(triangle[2]-triangle[0])*(j/n)[:,None])
        nearest = tree.query(grid, k=min(8, tree.n))[1]
        residual = residuals[np.asarray(nearest).reshape(len(grid), -1)]
        deviation.append(np.sqrt(np.mean(residual**2, axis=1)))
        vertices.append(grid)
        i,k=np.triu_indices(n)
        j=k-i
        start=i*(n+1)-i*(i-1)//2+j
        below=(i+1)*(n+1)-(i+1)*i//2+j
        first=np.column_stack((start,below,start+1))
        second=np.column_stack((below,below+1,start+1))
        keep=np.column_stack((np.ones(len(i),dtype=bool),j<n-i-1)).ravel()
        faces.append(np.stack((first,second),axis=1).reshape(-1,3)[keep]+offset)
        offset += len(grid)
    return (np.concatenate(vertices) if vertices else np.empty((0,3)),
            np.concatenate(faces) if faces else np.empty((0,3),dtype=np.int64),
            np.concatenate(deviation) if deviation else np.empty(0))


def corner_curves(result):
    """Final patch borders: analytic intersections, coplanar joins and outer rings."""
    curves = {"analytic": [], "coplanar": [], "outer": []}
    shared_segments = []
    for edge in result.shared_edges + result.bridges:
        kind = "analytic"
        if edge.get("kind") == "coplanar" or edge.get("bridge_type") == "coplanar":
            kind = "coplanar"
        elif edge.get("kind") == "bridge":
            a, b = [result.patches[int(label)] for label in edge["patch_ids"]]
            parallel = (a.kind == 'plane' and b.kind == 'plane' and
                        abs(a.normal @ b.normal) >= np.cos(np.deg2rad(10)))
            kind = "coplanar" if parallel else "outer"
        sources = list(edge["side_points"].values()) if "side_points" in edge else [edge.get("points", edge["endpoints"])]
        for source in sources:
            source = np.asarray(source)
            curves[kind].append(source)
            shared_segments.extend(zip(source[:-1], source[1:]))
    # Exterior boundaries come from the final triangles, so optimized/welded
    # borders are shown rather than the original alpha footprint.
    triangles = result.faces
    all_edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary = result.vertices[unique[counts == 1]]
    keep = np.ones(len(boundary), dtype=bool)
    if shared_segments:
        segments = np.asarray(shared_segments)
        start, delta = segments[:, 0], segments[:, 1]-segments[:, 0]
        length = np.maximum(np.einsum("ij,ij->i", delta, delta), 1e-20)
        for offset in range(0, len(boundary), 256):
            midpoints = boundary[offset:offset+256].mean(axis=1)
            v = midpoints[:, None, :]-start
            parameter = np.clip(np.einsum("ijk,jk->ij", v, delta)/length, 0, 1)
            distances = np.linalg.norm(v-parameter[:, :, None]*delta, axis=2)
            keep[offset:offset+len(midpoints)] = distances.min(axis=1) >= 1e-6
    curves["outer"].extend(boundary[keep])
    return curves


def point_triangle_distances(points, triangles):
    """Vectorized Euclidean distance to each corresponding final triangle."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b-a, c-a, points-a
    normal = np.cross(ab, ac)
    squared = np.einsum("ij,ij->i", normal, normal)
    projection = points - (np.einsum("ij,ij->i", ap, normal)/np.maximum(squared, 1e-30))[:, None]*normal
    v = projection-a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", v, ab)
    d21 = np.einsum("ij,ij->i", v, ac)
    denominator = np.maximum(d00*d11-d01*d01, 1e-30)
    u = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    inside = (u >= 0) & (w >= 0) & (u+w <= 1) & (squared > 1e-30)
    distances = np.full(len(points), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        t = np.clip(np.einsum("ij,ij->i", points-start, edge)/
                    np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(points-(start+t[:, None]*edge), axis=1))
    distances[inside] = np.linalg.norm(points[inside]-projection[inside], axis=1)
    return distances
