"""Open-shell reconstruction of planes, cylinders and PCA-local B-splines.

Dependencies: numpy scipy shapely>=2.1 (GEOS>=3.10) plyfile, geon's OCCT extension.
Primitive fitting and mesh stitching live in this package.

    from geon.algorithms.reconstruction.core import Config, reconstruct
    result = reconstruct(points, patch_ids, config=Config(), verbose=True)
    result.export_ply('reconstructed.ply')

Patch IDs remain separate. Optional part IDs restrict adjacency. Cylinder fitting
is selected only after a substantial residual improvement and angular support.
Surface intersections are preferred; bounded bridge strips can recover supported seams
without changing source primitive fidelity. Bridge/cap triangles belong to the
smaller original patch by point count. Intentional outer borders and single-patch
holes remain open. This does not promise a watertight solid or exact CAD B-rep.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
from time import perf_counter
import numpy as np
from scipy.spatial import Delaunay, cKDTree, QhullError
import shapely
from shapely.geometry import Polygon, LineString
from shapely.errors import GEOSException
from .stitching import stitch_boundaries
from .surfaces import Plane, fit_bspline
from .geometry_backend import triple_points
from .primitives import (fit_cylinder, curve_in_patch, primitive_residual,
    raw_intersection_curves, simplify_curve_3d, refine_surface_triangles)


@dataclass
class Config:
    min_points: int = 51  # Original retained patch size, before subsampling.
    subsample_distance: float | None = None  # Minimum spacing; None disables.
    exclude_ids: tuple[int, ...] = (-1,)
    alpha_radius: float = .10  # Circumradius, NOT inverse alpha.
    max_plane_rmse: float = .025
    fit_bsplines: bool = True
    max_bspline_rmse: float = .025
    bspline_error_ratio: float = .5
    bspline_smoothing: float = .001
    bspline_max_slope: float = 1.
    bspline_mesh_spacing: float = .5  # Maximum regular UV cell width, coordinate units.
    fit_cylinders: bool = True
    cylinder_trim_mode: str = "observed"  # "extent" regularizes complete columns.
    cylinder_plane_trigger: float = .02
    cylinder_error_ratio: float = .5
    max_cylinder_rmse: float = .025
    cylinder_max_radius: float = 10.
    cylinder_min_angle_degrees: float = 35.
    curve_tolerance: float = .005
    cylinder_chord_tolerance: float = .01
    cylinder_mesh_spacing: float = .25  # Maximum physical UV cell width.
    rescue_shared_edges: bool = True
    bridge_remaining_gaps: bool = True
    clean_border_pass: bool = True
    clean_first: bool = False  # Experimental coordinated joints; opt in for comparison.
    stitch_mesh_boundaries: bool = True
    mesh_stitch_tolerance: float = .05
    junction_cap_tolerance: float = .15
    gap_bridge_tolerance: float = .08
    regularize_coplanar_borders: bool = True
    adjacency_radius: float = .08
    min_contact_points: int = 5  # Unique points on EACH patch.
    edge_tolerance: float = .08
    support_step: float = .02
    join_coplanar: bool = True
    prefer_coplanar_join: bool = False
    coplanar_join_mode: str = "bridge"  # "bridge" preserves planes; "snap" shares a seam.
    coplanar_angle_degrees: float = 5.
    coplanar_plane_tolerance: float = .01  # Maximum local plane offset, metres.
    coplanar_boundary_tolerance: float = .08  # Maximum observed border separation.
    corner_tolerance: float | None = None  # Endpoint movement; None uses edge_tolerance.
    min_edge_length: float = .10
    min_plane_angle_degrees: float = 5.
    simplify_tolerance: float = .015
    weld_tolerance: float = 1e-7

    def __post_init__(self):
        for name in ('max_bspline_rmse', 'bspline_smoothing', 'bspline_max_slope', 'bspline_mesh_spacing', 'cylinder_mesh_spacing'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.bspline_error_ratio < 1:
            raise ValueError('bspline_error_ratio must be between 0 and 1')
        for name in ('alpha_radius', 'max_plane_rmse', 'adjacency_radius',
                     'edge_tolerance', 'support_step', 'min_edge_length',
                     'cylinder_plane_trigger', 'max_cylinder_rmse', 'cylinder_max_radius',
                     'curve_tolerance', 'cylinder_chord_tolerance', 'gap_bridge_tolerance',
                     'weld_tolerance', 'coplanar_plane_tolerance', 'coplanar_boundary_tolerance'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.corner_tolerance is not None and (not np.isfinite(self.corner_tolerance) or self.corner_tolerance <= 0):
            raise ValueError("corner_tolerance must be finite and positive, or None")
        if self.subsample_distance is not None and (not np.isfinite(self.subsample_distance) or self.subsample_distance <= 0):
            raise ValueError("subsample_distance must be finite and positive, or None")
        if self.cylinder_trim_mode not in ('observed','extent'):
            raise ValueError('cylinder_trim_mode must be observed or extent')
        if not 0 < self.cylinder_error_ratio < 1 or not 0 < self.cylinder_min_angle_degrees < 360:
            raise ValueError("Invalid cylinder selection thresholds")
        for name in ('mesh_stitch_tolerance','junction_cap_tolerance'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.coplanar_join_mode not in ("bridge", "snap"):
            raise ValueError("coplanar_join_mode must be bridge or snap")
        if not 0 < self.coplanar_angle_degrees < 90:
            raise ValueError('coplanar_angle_degrees must be between 0 and 90')
        if self.min_points < 3 or self.min_contact_points < 1:
            raise ValueError('Invalid minimum point count')
        if self.simplify_tolerance < 0 or not 0 < self.min_plane_angle_degrees < 90:
            raise ValueError('Invalid simplification tolerance or plane angle')


@dataclass
class Patch:
    id: int
    part_id: int | None
    center: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    points: np.ndarray
    footprint: object
    polygon: object
    rmse: float
    locked: list = field(default_factory=list)
    cylinder: object | None = None
    plane_rmse: float = 0.
    source_point_count: int = 0  # After input filtering, before subsampling.
    spline: object | None = None

    @property
    def kind(self):
        return self.surface.kind

    @property
    def surface(self):
        if self.spline is not None:
            return self.spline
        if self.cylinder is not None:
            return self.cylinder
        return Plane(self.center, self.normal, self.basis)

    def normal_at(self, uv):
        return self.surface.normal_at(uv)

    @property
    def parameter_bounds(self):
        if self.footprint is not None:
            return np.asarray(self.footprint.bounds)
        uv = self.project(self.points)
        return np.r_[uv.min(axis=0), uv.max(axis=0)]

    def residual(self, xyz):
        return self.surface.residual(xyz)

    def project(self, xyz):
        return self.surface.project(xyz)

    def lift(self, uv):
        return self.surface.lift(uv)


@dataclass
class Result:
    vertices: np.ndarray
    faces: np.ndarray
    face_patch_ids: np.ndarray
    face_part_ids: np.ndarray
    patches: dict
    shared_edges: list
    diagnostics: list
    bridges: list = field(default_factory=list)
    face_is_bridge: np.ndarray | None = None

    def export_ply(self, path):
        """Write float64 coordinates, deterministic patch colors, and face labels."""
        from plyfile import PlyData, PlyElement
        vertices = np.empty(len(self.vertices), dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        for i, name in enumerate(('x', 'y', 'z')):
            vertices[name] = self.vertices[:, i]
        faces = np.empty(len(self.faces), dtype=[('vertex_indices', 'i4', (3,)),
                         ('patch_id', 'i4'), ('part_id', 'i4'), ('is_bridge', 'u1'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        faces['vertex_indices'] = self.faces
        faces['patch_id'] = self.face_patch_ids
        faces['part_id'] = self.face_part_ids
        faces['is_bridge'] = 0 if self.face_is_bridge is None else self.face_is_bridge
        colors = np.random.default_rng(42).integers(40, 240, (len(self.patches), 3))
        palette = dict(zip(self.patches, colors))
        for i, label in enumerate(self.face_patch_ids):
            for j, name in enumerate(('red', 'green', 'blue')):
                faces[name][i] = palette[label][j]
        PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(faces, 'face')]).write(str(path))


def polygons(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for g in geometry.geoms:
            yield from polygons(g)


def spatial_subsample(points, min_distance):
    """Deterministic greedy minimum-distance sampling, retaining original points.

    KD-tree radius queries run in compiled SciPy; only selected points initiate
    queries. The first available point in input order wins. Coordinates remain
    float64; no Geon/native-extension dependency is required.
    """
    tree = cKDTree(points)
    available = np.ones(len(points), dtype=bool)
    keep = []
    for index in range(len(points)):
        if available[index]:
            keep.append(index)
            available[tree.query_ball_point(points[index], min_distance)] = False
    return points[np.asarray(keep, dtype=np.intp)]


def alpha_footprint(uv, radius, log=None):
    """Build polygons from exposed Delaunay edges, without triangle union.

    Polygonization can also produce faces filling holes. Classify each polygon
    using the retained triangle attached to one of its exposed boundary edges.
    This preserves nested islands and disconnected components without guessing
    ring orientation or walking ambiguous point-touching vertices ourselves.
    """
    started = perf_counter()
    def report(stage):
        nonlocal started
        now = perf_counter()
        if log is not None:
            log(f"Alpha {stage}: {now-started:.3f}s")
        started = now

    uv = np.unique(uv, axis=0)
    report(f"deduplicate ({len(uv):,} points)")
    triangulation = Delaunay(uv)
    triangles = triangulation.simplices
    report(f"Delaunay ({len(triangles):,} triangles)")
    p = uv[triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area2 = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
    numerator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)*np.linalg.norm(a-b, axis=1)
    radii = np.full(len(p), np.inf)
    np.divide(numerator, 2*area2, out=radii, where=area2 > 0)
    retained = radii <= radius
    kept = triangles[retained]
    if not len(kept):
        raise ValueError('Alpha radius retains no triangles')
    report(f"circumradius filter ({len(kept):,} retained)")

    edges = np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]])
    edges.sort(axis=1)
    unique_edges, first, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_triangles = kept[first[counts == 1] % len(kept)]
    report(f"boundary edge counting ({len(boundary_edges):,} edges)")

    candidates = list(shapely.polygonize(shapely.linestrings(uv[boundary_edges])).geoms)
    if not candidates:
        raise ValueError('Alpha boundary contains no polygon loops')
    # find_simplex lazily computes transforms for EVERY Delaunay triangle. On
    # million-triangle scans that costs tens of seconds to classify a few loops.
    # Each exposed edge has exactly one retained incident triangle. Its centroid
    # lies inside the polygon on the material side, and outside the hole face.
    witnesses={tuple(uv[edge].ravel()):uv[triangle].mean(axis=0)
               for edge,triangle in zip(boundary_edges,boundary_triangles)}
    selected=[]
    for poly in candidates:
        edge=np.asarray(poly.exterior.coords)[:2]
        if tuple(edge[0])>tuple(edge[1]):
            edge=edge[::-1]
        witness=shapely.Point(witnesses[tuple(edge.ravel())])
        if poly.covers(witness):
            selected.append(poly)
    if not selected:
        raise ValueError('Alpha boundary contains no retained polygon faces')
    result = selected[0] if len(selected) == 1 else shapely.MultiPolygon(selected)
    if not result.is_valid:
        raise ValueError('Alpha boundary polygonization produced invalid geometry')
    report(f"polygonize/classify ({len(selected):,} components)")
    return result


def _rings(geometry):
    for poly in polygons(geometry):
        yield np.asarray(poly.exterior.coords)[:-1]
        for ring in poly.interiors:
            yield np.asarray(ring.coords)[:-1]


def _boundary_samples(patch, step):
    chunks = []
    for ring in _rings(patch.footprint):
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            n = max(1, int(np.ceil(np.linalg.norm(b-a)/step)))
            chunks.append(a + np.arange(n)[:, None]/n*(b-a))
    return patch.lift(np.concatenate(chunks))


def _candidate_pairs(patches, cfg, log=lambda message: None):
    """AABB broad phase, then bounded-memory nearest-neighbor queries in C++."""
    radius = max(cfg.adjacency_radius, cfg.coplanar_boundary_tolerance) if cfg.join_coplanar else cfg.adjacency_radius
    ids = list(patches)
    bounds = {i: (p.points.min(axis=0), p.points.max(axis=0)) for i, p in patches.items()}
    trees = {i: cKDTree(p.points) for i, p in patches.items()}
    frames = {i: np.column_stack([p.basis,p.normal]) for i,p in patches.items()}
    local_bounds = {}
    for i,p in patches.items():
        local=(p.points-p.center)@frames[i]
        local_bounds[i]=(local.min(axis=0)-radius-1e-10,local.max(axis=0)+radius+1e-10)
    def possible_contacts(points, target):
        projected=(points-patches[target].center)@frames[target]
        lo,hi=local_bounds[target]
        return points[np.all((projected>=lo)&(projected<=hi),axis=1)]
    # Sweep on x avoids examining every pair when patches are spatially separated.
    order = sorted(ids, key=lambda i: bounds[i][0][0])
    for pos, i in enumerate(order):
        log(f"Adjacency scan {pos + 1}/{len(order)}: patch {i}")
        a = patches[i]
        for j in order[pos+1:]:
            if bounds[j][0][0] > bounds[i][1][0] + radius:
                break
            b = patches[j]
            if a.part_id != b.part_id:
                continue
            gap = np.maximum(0, np.maximum(bounds[i][0]-bounds[j][1], bounds[j][0]-bounds[i][1]))
            if np.linalg.norm(gap) > radius:
                continue
            aa = possible_contacts(a.points,j)
            if len(aa)<cfg.min_contact_points:
                continue
            da = trees[j].query(aa, distance_upper_bound=radius)[0]
            if np.count_nonzero(np.isfinite(da)) < cfg.min_contact_points:
                continue
            bb = possible_contacts(b.points,i)
            if len(bb)<cfg.min_contact_points:
                continue
            db = trees[i].query(bb, distance_upper_bound=radius)[0]
            if np.count_nonzero(np.isfinite(db)) >= cfg.min_contact_points:
                yield i, j


def _intersection_segments(a, b, cfg, diagnostic=None):
    report = diagnostic or (lambda message: None)
    if a.kind == 'plane' and b.kind == 'plane':
        if np.linalg.norm(np.cross(a.normal, b.normal)) < 1e-8 and abs((b.center-a.center)@a.normal) < 1e-7:
            report('coincident planes: defer to supported coplanar border joining')
            return []
        if cfg.join_coplanar and cfg.prefer_coplanar_join and abs(a.normal@b.normal)>=np.cos(np.deg2rad(cfg.coplanar_angle_degrees)):
            return []
        if np.linalg.norm(np.cross(a.normal, b.normal)) < np.sin(np.deg2rad(cfg.min_plane_angle_degrees)):
            return []
    trees = [cKDTree(_boundary_samples(p, cfg.support_step)) for p in (a, b)]
    point_trees = [cKDTree(p.points) for p in (a, b)]
    result = []
    for raw in raw_intersection_curves(a, b, cfg.support_step, cfg.curve_tolerance, cfg.edge_tolerance, diagnostic=report):
        finite = np.isfinite(raw).all(axis=1)
        valid = finite.copy()
        for tree, point_tree in zip(trees, point_trees):
            distances = np.full(len(raw), np.inf)
            distances[finite] = tree.query(raw[finite])[0]
            valid &= distances <= cfg.edge_tolerance
            distances[finite] = point_tree.query(raw[finite])[0]
            valid &= distances <= cfg.adjacency_radius
        closed = finite.all() and np.linalg.norm(raw[0]-raw[-1]) < cfg.support_step*.1
        runs = _cyclic_supported_runs(valid) if closed else [x for x in np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1)+1) if len(x)]
        for indices in runs:
            curve = raw[indices]
            if len(curve) < 2 or np.linalg.norm(np.diff(curve, axis=0), axis=1).sum() < cfg.min_edge_length:
                continue
            cylinders = [p.cylinder for p in (a,b) if p.cylinder is not None]
            tolerance = cfg.curve_tolerance
            if cylinders:
                tolerance = min(tolerance, cfg.cylinder_chord_tolerance*.5, min(c.radius for c in cylinders)*.02)
            curve = simplify_curve_3d(curve, tolerance)
            result.append(curve)
    return [piece for curve in result for piece in _supported_boundary_pieces(a,b,curve,cfg)]


def _edge_points(edge):
    return edge.get('points', edge['endpoints'])


def _has_curve(geometry, points):
    return _has_curves(geometry, [points])


def _has_edge(geometry, endpoints, atol=1e-8):
    return _has_curves(geometry, [endpoints], atol)


def _preserves_local_locks(patch, geometry, curve, tolerance):
    """Only a chain inside the new curve's tolerance box can be replaced.

    _batch_boundary checks every removed chain vertex against that curve.
    Existing locks outside this box therefore remain unchanged by construction.
    """
    if not patch.locked:return True
    locked=np.asarray(patch.locked)
    low=curve.min(axis=0)-tolerance-1e-8
    high=curve.max(axis=0)+tolerance+1e-8
    near=np.all(locked.max(axis=1)>=low,axis=1)&np.all(locked.min(axis=1)<=high,axis=1)
    return _has_curves(geometry,locked[near])


def _has_curves(geometry, curves, atol=1e-8):
    """Check complete boundary segments in one spatial index, in either direction.

    Repeated Python scans of every ring for every locked seam made long girder
    boundary recovery quadratic in the number of accumulated seam segments.
    """
    queries = [np.stack((c[:-1],c[1:]),axis=1) for c in map(np.asarray,curves) if len(c)>1]
    if not queries:
        return True
    segments = [np.stack((r,np.roll(r,-1,axis=0)),axis=1) for r in _rings(geometry) if len(r)]
    if not segments:
        return False
    segments = np.concatenate(segments)
    segments = np.concatenate((segments,segments[:,::-1]))
    queries = np.concatenate(queries)
    tree=cKDTree(segments.reshape(-1,4))
    _,nearest=tree.query(queries.reshape(-1,4))
    exact=np.all(np.linalg.norm(segments[nearest]-queries,axis=2)<atol,axis=1)
    missing=queries[~exact]
    if not len(missing):
        return True
    neighbors=tree.query_ball_point(missing.reshape(-1,4),np.nextafter(np.sqrt(2)*atol,np.inf))
    missing=[query for query,ids in zip(missing,neighbors)
             if not (len(ids) and np.any(np.all(np.linalg.norm(segments[ids]-query,axis=2)<atol,axis=1)))]
    if not len(missing):return True
    # Inserting a junction may subdivide an existing seam without moving it.
    # Test complete collinear interval coverage, not just matching endpoints.
    lines=shapely.linestrings(segments)
    index=shapely.STRtree(lines)
    for query in missing:
        delta=query[1]-query[0];length2=float(delta@delta)
        if length2<=atol*atol:
            continue
        ids=index.query(LineString(query),predicate='dwithin',distance=atol)
        candidates=segments[ids]
        t=np.einsum('nki,i->nk',candidates-query[0],delta)/length2
        distance=np.linalg.norm(candidates-query[0]-t[:,:,None]*delta,axis=2)
        intervals=np.sort(t[np.all(distance<=atol,axis=1)],axis=1)
        end=0.;fraction=atol/np.sqrt(length2)
        for lo,hi in sorted(intervals.tolist()):
            if lo>end+fraction:break
            end=max(end,hi)
        if end<1-fraction:return False
    return True


def _reconcile_corners(patches, proposals, cfg, log):
    """Snap nearby edge endpoints to supported, well-conditioned three-surface corners.

    No averaging across incompatible surfaces. All displacement checks use the
    original endpoints; an endpoint is assigned at most once (no chain drift).
    """
    from itertools import combinations
    tolerance = cfg.edge_tolerance if cfg.corner_tolerance is None else cfg.corner_tolerance
    incident = {i: [] for i in patches}
    for k, edge in enumerate(proposals):
        edge['endpoints'] = edge['endpoints'].copy()
        for i in edge['patch_ids']:
            incident[i].append(k)
    triples = set()
    for indices in incident.values():
        for a, b in combinations(indices, 2):
            ids = tuple(sorted(set(proposals[a]['patch_ids']) | set(proposals[b]['patch_ids'])))
            if len(ids) == 3:
                triples.add(ids)
    trees = {i: cKDTree(p.points) for i, p in patches.items()}
    candidates = []
    for ids in sorted(triples):
        ps = [patches[i] for i in ids]
        try:
            corners = triple_points(ps, tolerance)
        except RuntimeError as exc:
            log(f'Corner {ids}: {exc}')
            continue
        for corner in corners:
            matrix = np.stack([p.normal_at(p.project(corner)) for p in ps])
            if np.linalg.cond(matrix) > 1000:
                continue
            if any(trees[p.id].query(corner)[0] > cfg.adjacency_radius or
                   p.footprint.boundary.distance(shapely.Point(p.project(corner))) > cfg.edge_tolerance
                   for p in ps):
                continue
            endpoints = []
            for k in sorted(set().union(*(set(incident[i]) for i in ids))):
                if not set(proposals[k]['patch_ids']).issubset(ids):
                    continue
                distances = np.linalg.norm(proposals[k]['endpoints']-corner, axis=1)
                end = int(np.argmin(distances))
                if distances[end] <= tolerance:
                    endpoints.append((k, end, float(distances[end])))
            if len(endpoints) >= 2 and len(set().union(*(set(proposals[k]['patch_ids']) for k, _, _ in endpoints))) == 3:
                candidates.append((max(d for _, _, d in endpoints), ids, corner, endpoints))
    assigned = set()
    accepted = 0
    for _, ids, corner, endpoints in sorted(candidates, key=lambda x: (x[0], x[1])):
        if any((k, end) in assigned for k, end, _ in endpoints):
            continue
        # Do not collapse or reverse short edges while moving their endpoints.
        if any(np.linalg.norm(proposals[k]['endpoints'][1-end]-corner) < cfg.min_edge_length
               or np.dot(proposals[k]['endpoints'][1-end]-corner,
                         proposals[k]['endpoints'][1-end]-proposals[k]['endpoints'][end]) <= 0
               for k, end, _ in endpoints):
            continue
        for k, end, _ in endpoints:
            proposals[k]['endpoints'][end] = corner.copy()
            assigned.add((k, end))
        accepted += 1
    for edge in proposals:
        if 'points' in edge:
            edge['points'][0] = edge['endpoints'][0]
            edge['points'][-1] = edge['endpoints'][-1]
    log(f'Corner reconciliation: {accepted} supported corners, {len(assigned)} endpoint assignments')
    return accepted


def _insert_boundary_endpoints(ring, endpoints, tolerance):
    """Split original ring edges at closest projections of analytical endpoints.

    New vertices initially stay on the observed boundary. Chain replacement then
    moves them onto the exact analytical endpoints. No existing alpha vertex is
    required near a proposed endpoint, and shared corners use identical targets.
    """
    n = len(ring)
    vectors = np.roll(ring, -1, axis=0)-ring
    lengths2 = np.einsum('ij,ij->i', vectors, vectors)
    parameters = list(np.arange(n, dtype=float))
    for endpoint in endpoints:
        fraction = np.zeros(n)
        np.divide(np.einsum('ij,ij->i', endpoint-ring, vectors), lengths2,
                  out=fraction, where=lengths2 > 0)
        fraction = np.clip(fraction, 0., 1.)
        projected = ring+fraction[:, None]*vectors
        distances = np.linalg.norm(projected-endpoint, axis=1)
        edge = int(np.argmin(distances))
        if distances[edge] <= tolerance:
            parameter = (edge+fraction[edge]) % n
            # Avoid microscopic sliver edges at existing vertices.
            if abs(parameter-round(parameter)) < 1e-10:
                parameter = float(round(parameter) % n)
            parameters.append(parameter)
    parameters = np.sort(parameters)
    parameters = parameters[np.r_[True, np.diff(parameters) > 1e-10]]
    edges = np.floor(parameters).astype(int)
    fractions = parameters-edges
    return ring[edges]+fractions[:, None]*vectors[edges]


def _chain_options(ring, endpoints, tolerance, check_distance=True):
    """Candidate arcs after exact endpoint projections split boundary edges."""
    n = len(ring)
    if np.linalg.norm(endpoints[0]-endpoints[-1]) < 1e-9:
        line = LineString(endpoints)
        if len(endpoints) < 4 or np.max(shapely.distance(shapely.points(ring), line)) > tolerance:
            return []
        return [(line.length, 0, 0, endpoints, set(range(n)))]
    start, stop = [int(np.argmin(np.linalg.norm(ring-e, axis=1))) for e in endpoints[[0, -1]]]
    if start == stop or any(np.linalg.norm(ring[k]-e) > tolerance for k, e in zip((start, stop), endpoints[[0, -1]])):
        return []
    line = LineString(endpoints)
    options = []
    for first, last, ends in ((start, stop, endpoints), (stop, start, endpoints[::-1])):
        steps = (last-first) % n
        indices = (np.arange(steps+1)+first) % n
        if check_distance and np.max(shapely.distance(shapely.points(ring[indices]), line)) > tolerance:
            continue
        length = np.linalg.norm(np.diff(ring[indices], axis=0), axis=1).sum()
        options.append((length, first, last, ends, set(indices[:-1].tolist())))
    return sorted(options, key=lambda x: x[0])


def _batch_boundary(patch, edges, cfg):
    """Rebuild all proposed chains together from the original alpha footprint.

    Analytical endpoint projections split boundary segments before chain matching.
    Adjacent chains may share a split vertex if they agree on its new corner.
    Overlapping chain interiors or inconsistent endpoint targets are rejected.
    """
    original = list(polygons(patch.footprint))
    rings = []
    for pi, poly in enumerate(original):
        for ri, ring in enumerate([poly.exterior, *poly.interiors]):
            rings.append((pi, ri, np.asarray(ring.coords)[:-1]))
    targets = np.concatenate([curve_in_patch(patch, _edge_points(e))[[0, -1]] for e in edges]) if edges else np.empty((0, 2))
    insertion_tolerance = max([cfg.edge_tolerance]+[
        cfg.coplanar_boundary_tolerance for e in edges if e.get('kind') == 'coplanar'])
    rings = [(pi, ri, _insert_boundary_endpoints(ring, targets, insertion_tolerance))
             for pi, ri, ring in rings]
    edits = {k: [] for k in range(len(rings))}
    for edge in edges:
        endpoints = curve_in_patch(patch, _edge_points(edge))
        options = []
        for k, (_, _, ring) in enumerate(rings):
            tolerance = cfg.coplanar_boundary_tolerance if edge.get('kind') == 'coplanar' else cfg.edge_tolerance
            for option in _chain_options(ring, endpoints, tolerance):
                options.append((option[0], k, option))
        placed = False
        for _, k, option in sorted(options, key=lambda x: x[0]):
            _, first, last, ends, covered = option
            targets = {first: ends[0], last: ends[-1]}
            compatible = True
            for previous in edits[k]:
                _, f, l, e, used = previous
                if covered & used:
                    compatible = False
                    break
                for vertex, position in ((f, e[0]), (l, e[-1])):
                    if vertex in targets and np.linalg.norm(targets[vertex]-position) > 1e-8:
                        compatible = False
                        break
            if compatible:
                edits[k].append(option)
                placed = True
                break
        if not placed:
            return None
    rebuilt = {}
    for k, (pi, ri, ring) in enumerate(rings):
        if not edits[k]:
            rebuilt[pi, ri] = ring
            continue
        events = {e[1]: e for e in edits[k]}
        start = edits[k][0][1]
        i, consumed, output = start, 0, []
        while consumed < len(ring):
            if i in events:
                _, first, last, ends, _ = events[i]
                output.extend(ends)
                steps = len(events[i][4])
                consumed += steps
                i = last
            else:
                output.append(ring[i])
                consumed += 1
                i = (i+1) % len(ring)
        output = np.asarray(output)
        output = output[np.r_[True, np.linalg.norm(np.diff(output, axis=0), axis=1) > 1e-10]]
        if len(output) > 1 and np.linalg.norm(output[-1]-output[0]) < 1e-10:
            output = output[:-1]
        if len(output) < 3:
            return None
        rebuilt[pi, ri] = output
    result = []
    for pi, poly in enumerate(original):
        candidate = Polygon(rebuilt[pi, 0], [rebuilt[pi, j+1] for j in range(len(poly.interiors))])
        if not candidate.is_valid or candidate.area <= 0:
            return None
        result.append(candidate)
    geometry = result[0] if len(result) == 1 else shapely.MultiPolygon(result)
    if getattr(patch, 'spline', None) is not None:
        # Coplanar/gap recovery tolerances can exceed edge_tolerance. They must
        # never extend a height patch past its fitted, bounded border halo.
        bounds = np.asarray(geometry.bounds)
        domain = patch.spline.bounds
        if np.any(bounds[:2] < domain[:2]-1e-8) or np.any(bounds[2:] > domain[2:]+1e-8):
            return None
    if not geometry.is_valid or not _has_curves(geometry, [curve_in_patch(patch, _edge_points(e)) for e in edges]):
        return None
    return geometry


def _supported_boundary_pieces(a, b, curve, cfg):
    """Split a seam at deep observed notches; avoid crossing retained end borders.

    A nearest-boundary radius test alone can bridge a narrow, deep notch: each
    seam sample is close to a notch mouth although its boundary chain dives far
    away. Keep those gaps, then trim only failing endpoints by a bounded amount.
    """
    def valid(points):
        edge = dict(patch_ids=(a.id,b.id),points=points,endpoints=points[[0,-1]])
        return all(_batch_boundary(p,[edge],cfg) is not None for p in (a,b))
    if valid(curve):
        return [curve]
    if np.linalg.norm(curve[0]-curve[-1]) < 1e-8:
        return [curve]  # Closed-ring conflicts are handled jointly by assembly.
    distances = np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
    blocked = []
    for p in (a,b):
        uv = curve_in_patch(p,curve)
        line = LineString(uv)
        options = []
        for ring in _rings(p.footprint):
            ring = _insert_boundary_endpoints(ring,uv[[0,-1]],cfg.edge_tolerance)
            for option in _chain_options(ring,uv,cfg.edge_tolerance,check_distance=False):
                _,first,last,_,_ = option
                indices = (np.arange((last-first)%len(ring)+1)+first)%len(ring)
                chain = ring[indices]
                gap = shapely.distance(shapely.points(chain),line)
                options.append((float(np.mean(gap)),chain,gap))
        if not options:
            continue
        _,chain,gap = min(options,key=lambda x:x[0])
        bad = np.flatnonzero(gap > cfg.edge_tolerance)
        positions = shapely.line_locate_point(line,shapely.points(chain))
        uv_distances = np.r_[0,np.cumsum(np.linalg.norm(np.diff(uv,axis=0),axis=1))]
        positions = np.interp(positions,uv_distances,distances)
        for run in np.split(bad,np.flatnonzero(np.diff(bad)>1)+1):
            if not len(run):
                continue
            lo,hi=max(0,run[0]-1),min(len(chain)-1,run[-1]+1)
            loc = positions[lo:hi+1]
            blocked.append((max(0,float(loc.min())-cfg.support_step),
                            min(distances[-1],float(loc.max())+cfg.support_step)))
    intervals, start = [], 0.
    for lo,hi in sorted(blocked):
        if lo>start:
            intervals.append((start,lo))
        start=max(start,hi)
    intervals.append((start,distances[-1]))

    def section(lo,hi):
        params=np.r_[lo,distances[(distances>lo)&(distances<hi)],hi]
        return np.column_stack([np.interp(params,distances,curve[:,k]) for k in range(3)])
    pieces=[]
    for lo,hi in intervals:
        if hi-lo < cfg.min_edge_length:
            continue
        # Most failures occur where moving an endpoint makes a short loop with
        # an untouched border. Try each end independently; never move farther
        # than the configured boundary tolerance or fill an unsupported gap.
        trims = sorted(set([0., *[min(cfg.edge_tolerance,cfg.support_step*2**i) for i in range(5)]]))
        attempts = sorted(((x,y) for x in trims for y in trims),key=lambda v:(sum(v),max(v),v))
        accepted=None
        for left,right in attempts:
            if hi-lo-left-right < cfg.min_edge_length:
                continue
            part=section(lo+left,hi-right)
            if valid(part):
                accepted=part
                break
        pieces.append(section(lo,hi) if accepted is None else accepted)
    return pieces


def _assemble_topology(patches, proposals, cfg, diagnostics, log):
    """Try simultaneous edits first; prune incompatible proposals globally.

    An edge is retained only when BOTH faces can represent it. Rebuild from the
    original footprints every time, so no early endpoint is permanently locked.
    """
    active = list(proposals)
    # Remove conflicts, retaining the largest feasible prefix per face. Subsequent
    # rounds rebuild both sides after any removals; shared corners stay identical.
    while active:
        rejected = set()
        for patch in patches.values():
            indices = [k for k, e in enumerate(active) if patch.id in e['patch_ids']]
            edges = [active[k] for k in indices]
            if not edges or _batch_boundary(patch, edges, cfg) is not None:
                continue
            retained = []
            for k in sorted(indices, key=lambda k: -np.linalg.norm(active[k]['endpoints'][1]-active[k]['endpoints'][0])):
                if _batch_boundary(patch, retained+[active[k]], cfg) is not None:
                    retained.append(active[k])
                else:
                    rejected.add(k)
        if not rejected:
            break
        for k in sorted(rejected):
            diagnostics.append(f"Pair {active[k]['patch_ids']}: rejected incompatible batch boundary edit")
        log(f'Boundary assembly: rejected {len(rejected)} conflicting proposals; rebuilding both sides')
        active = [e for k, e in enumerate(active) if k not in rejected]
    for patch in patches.values():
        edges = [e for e in active if patch.id in e['patch_ids']]
        patch.polygon = _batch_boundary(patch, edges, cfg) if edges else patch.footprint
        patch.locked = [curve_in_patch(patch, curve)[k:k+2] for e in edges
                        for curve in [_edge_points(e)] for k in range(len(curve)-1)]
    log(f'Boundary assembly: accepted {len(active)}/{len(proposals)} shared segments')
    return active


def _rescue_edges(patches, proposals, shared, cfg, diagnostics, log):
    """Retry against already-aligned boundaries, preserving existing constraints."""
    if not cfg.rescue_shared_edges:
        return shared
    import copy
    active = list(shared)
    accepted_ids = {id(e) for e in active}
    count = 0
    for _ in range(3):
        recovered = 0
        for edge in proposals:
            if id(edge) in accepted_ids:
                continue
            edits = []
            for label in edge['patch_ids']:
                patch = patches[label]
                current = copy.copy(patch)
                current.footprint = patch.polygon
                geometry = _batch_boundary(current, [edge], cfg)
                if geometry is None or not _preserves_local_locks(
                        patch,geometry,curve_in_patch(patch,_edge_points(edge)),cfg.edge_tolerance):
                    break
                edits.append((patch, geometry))
            if len(edits) == 2:
                for patch, geometry in edits:
                    patch.polygon = geometry
                    uv = curve_in_patch(patch, _edge_points(edge))
                    patch.locked.extend(uv[k:k+2] for k in range(len(uv)-1))
                active.append(edge);accepted_ids.add(id(edge));recovered += 1
        count += recovered
        if not recovered:
            break
    if count:
        message = f'Boundary rescue: recovered {count} shared edges against aligned footprints'
        diagnostics.append(message);log(message)
    # Subsequent joint coplanar rebuilding must use these aligned source footprints.
    for patch in patches.values():
        patch.footprint = patch.polygon
    return active


def _sample_border_runs(patch, step):
    """Ordered boundary samples with world-space directions into the face."""
    runs = []
    for poly in polygons(patch.footprint):
        for is_hole, ring in [(False, poly.exterior)]+[(True, r) for r in poly.interiors]:
            uv = np.asarray(ring.coords)[:-1]
            delta = np.roll(uv, -1, axis=0)-uv
            area2 = np.sum(uv[:, 0]*np.roll(uv[:, 1], -1)-uv[:, 1]*np.roll(uv[:, 0], -1))
            sign = (1 if area2 > 0 else -1)*(-1 if is_hole else 1)
            samples, inward = [], []
            for point, vector in zip(uv, delta):
                length = np.linalg.norm(vector)
                if length <= 1e-12:
                    continue
                count = max(1, int(np.ceil(length/step)))
                sample = point+np.arange(count)[:, None]/count*vector
                samples.append(sample)
                direction = sign*np.array([-vector[1], vector[0]])/length
                eps = max(1e-7, step*1e-4)
                world = patch.lift(sample+direction*eps)-patch.lift(sample)
                world /= np.maximum(np.linalg.norm(world, axis=1)[:, None], 1e-15)
                inward.append(world)
            if samples:
                runs.append((patch.lift(np.concatenate(samples)), np.concatenate(inward)))
    return runs


def _cyclic_supported_runs(valid):
    if np.all(valid):
        return [np.arange(len(valid))]
    cut = int(np.flatnonzero(~valid)[0])
    order = (np.arange(len(valid))+cut+1) % len(valid)
    indices = np.flatnonzero(valid[order])
    return [order[x] for x in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(x)]


def _coplanar_curves(a, b, cfg, corner_gap=False, border_cache=None):
    """Match opposing local border runs; simplify their common midpoint once."""
    planar = a.kind == b.kind == 'plane'
    cosine = abs(float(a.normal @ b.normal))
    if planar and not corner_gap and cosine < np.cos(np.deg2rad(cfg.coplanar_angle_degrees)):
        return []
    if planar and corner_gap and cosine >= np.cos(np.deg2rad(cfg.coplanar_angle_degrees)):
        return []
    def runs(patch):
        if border_cache is None:
            return _sample_border_runs(patch,cfg.support_step)
        if patch.id not in border_cache:
            border_cache[patch.id]=_sample_border_runs(patch,cfg.support_step)
        return border_cache[patch.id]
    aruns,bruns = runs(a),runs(b)
    if not aruns or not bruns:
        return []
    bxyz = np.concatenate([x for x, _ in bruns])
    binward = np.concatenate([x for _, x in bruns])
    if b.locked:
        locked = shapely.MultiLineString(b.locked)
        exposed = shapely.distance(shapely.points(b.project(bxyz)),locked) > 1e-7
        bxyz, binward = bxyz[exposed], binward[exposed]
        if not len(bxyz):return []
    tree = cKDTree(bxyz)
    proposals = []
    for axyz, ainward in aruns:
        distance, nearest = tree.query(axyz)
        matched = bxyz[nearest]
        valid = distance <= (cfg.gap_bridge_tolerance if corner_gap else cfg.coplanar_boundary_tolerance)
        if a.locked:
            valid &= shapely.distance(shapely.points(a.project(axyz)),shapely.MultiLineString(a.locked)) > 1e-7
        if not np.any(valid):
            continue
        an = a.normal_at(a.project(axyz))
        bn = b.normal_at(b.project(matched))
        local_cos = np.abs(np.einsum('ij,ij->i', an, bn))
        parallel = local_cos >= np.cos(np.deg2rad(cfg.coplanar_angle_degrees))
        valid &= ~parallel if corner_gap else parallel
        if not np.any(valid):
            continue
        # Face interiors must lie on opposite sides of the seam. This excludes
        # duplicated/stacked same-side borders and transverse point contacts.
        if not corner_gap:
            valid &= np.einsum('ij,ij->i', ainward, binward[nearest]) < -.5
            # Nearest endpoints beyond the overlapping border range must not
            # extend the seam diagonally into either patch's outer boundary.
            gap = matched-axyz
            valid &= np.abs(np.einsum('ij,ij->i', gap, np.cross(ainward, an))) <= cfg.support_step*.55+1e-8
            valid &= np.abs(np.einsum('ij,ij->i', gap, np.cross(binward[nearest], bn))) <= cfg.support_step*.55+1e-8
            # OCCT closest-point solves are expensive on splines. Only solve
            # them for samples that passed all cheap border/normal predicates.
            indices=np.flatnonzero(valid)
            if not len(indices):
                continue
            valid[indices] &= np.abs(primitive_residual(b,axyz[indices])) <= cfg.coplanar_plane_tolerance
            indices=np.flatnonzero(valid)
            if not len(indices):
                continue
            valid[indices] &= np.abs(primitive_residual(a,matched[indices])) <= cfg.coplanar_plane_tolerance
        else:
            tangent_a = np.cross(ainward,an)
            tangent_b = np.cross(binward[nearest],bn)
            valid &= np.abs(np.einsum('ij,ij->i',tangent_a,tangent_b)) > .85
            gap = matched-axyz
            valid &= np.einsum('ij,ij->i',gap,ainward) <= distance*.15
            valid &= np.einsum('ij,ij->i',-gap,binward[nearest]) <= distance*.15
            if not np.any(valid):
                continue
            if planar:
                d = np.cross(a.normal,b.normal);d /= np.linalg.norm(d)
                origin = (a.center+b.center)/2
                p0 = origin+np.linalg.solve(np.stack([a.normal,b.normal,d]),[a.normal@(a.center-origin),b.normal@(b.center-origin),0.])
                for samples in (axyz,matched):
                    off=samples-p0;valid &= np.linalg.norm(off-(off@d)[:,None]*d,axis=1)<=cfg.edge_tolerance
            else:
                try:
                    curves = raw_intersection_curves(a,b,cfg.support_step,cfg.curve_tolerance,cfg.edge_tolerance)
                except RuntimeError:
                    curves = []
                if not curves:
                    continue
                supported = cKDTree(np.concatenate(curves))
                valid &= supported.query(axyz)[0] <= cfg.edge_tolerance
                valid &= supported.query(matched)[0] <= cfg.edge_tolerance
        for run in _cyclic_supported_runs(valid):
            if len(run) < 3 or np.linalg.norm(np.diff(axyz[run], axis=0), axis=1).sum() < cfg.min_edge_length:
                continue
            aside, bside = axyz[run], matched[run]
            common = (aside+bside)/2
            # Remove repeated nearest-neighbor-induced points before simplification.
            distinct = np.r_[True, np.linalg.norm(np.diff(common, axis=0), axis=1) > 1e-9]
            common, aside, bside = common[distinct], aside[distinct], bside[distinct]
            closed = len(run) == len(valid) and np.all(valid)
            if closed:
                common = np.vstack([common, common[0]])
                aside = np.vstack([aside, aside[0]])
                bside = np.vstack([bside, bside[0]])
            if len(common) < 2 or (not closed and np.linalg.norm(common[-1]-common[0]) < cfg.min_edge_length):
                continue
            uv = a.project(common)
            if not LineString(uv).is_simple:
                continue
            if planar:
                simplified = np.asarray(LineString(uv).simplify(cfg.simplify_tolerance).coords)
                keep = cKDTree(uv).query(simplified)[1]
            else:
                simplified = simplify_curve_3d(common, min(cfg.simplify_tolerance, cfg.curve_tolerance))
                keep = cKDTree(common).query(simplified)[1]
            common = common[keep]
            if len(common) >= 2:
                proposals.append({'patch_ids': (a.id, b.id), 'kind': 'coplanar',
                                  'points': common, 'endpoints': common[[0, -1]],
                                  'side_points': {a.id: aside[keep], b.id: bside[keep]},
                                  'bridge_type': 'corner_gap' if corner_gap else ('coplanar' if planar else 'curved')})
    return proposals


def _consistent_vertex_targets(patch, edges, tolerance):
    curves = [curve_in_patch(patch, _edge_points(e)) for e in edges]
    if not curves:
        return True
    uv = np.concatenate(curves)
    xyz = np.concatenate([_edge_points(e) for e in edges])
    pairs = cKDTree(uv).query_pairs(1e-8, output_type='ndarray')
    return not len(pairs) or np.all(np.linalg.norm(xyz[pairs[:, 0]]-xyz[pairs[:, 1]], axis=1) <= tolerance)


def _local_edge(edge, label):
    if edge.get('kind') != 'bridge':
        return edge
    curve = edge['side_points'][label]
    return dict(edge, kind='coplanar', points=curve, endpoints=curve[[0, -1]])


def _mesh_edge_uv(patch,edge):
    if patch.id in edge.get('mesh_uv',{}):
        return edge['mesh_uv'][patch.id]
    return curve_in_patch(patch,_edge_points(edge))


def _synchronize_existing_seam_nodes(edges, patches):
    """Share every boundary subdivision across both sides of a seam."""
    nodes={label:np.concatenate(list(_rings(p.polygon))) for label,p in patches.items()}
    for edge in edges:
        charts={label:curve_in_patch(patches[label],_edge_points(_local_edge(edge,label)))
                for label in edge['patch_ids']}
        fractions=[{0.,1.} for _ in range(len(_edge_points(edge))-1)]
        for label,curve in charts.items():
            for k,(a,b) in enumerate(zip(curve[:-1],curve[1:])):
                delta=b-a;length2=float(delta@delta)
                if length2<=1e-20:continue
                t=(nodes[label]-a)@delta/length2
                near=(t>1e-9)&(t<1-1e-9)
                near &= np.linalg.norm(nodes[label]-a-t[:,None]*delta,axis=1)<=1e-8
                fractions[k].update(np.round(t[near],12).tolist())
        def split(curve):
            return np.vstack([a+np.array(sorted(ts))[:-1,None]*(b-a)
                              for a,b,ts in zip(curve[:-1],curve[1:],fractions)]+[curve[-1:]])
        edge['points']=split(_edge_points(edge))
        edge['endpoints']=edge['points'][[0,-1]].copy()
        if 'side_points' in edge:
            edge['side_points']={label:split(curve) for label,curve in edge['side_points'].items()}
        edge['mesh_uv']={label:split(curve) for label,curve in charts.items()}


def _bridge_is_valid(edge, patches, geometries, cfg):
    """Reject crossing strips and strips that overlap either edited face."""
    a, b = [patches[i] for i in edge['patch_ids']]
    aa, bb = [edge['side_points'][i] for i in edge['patch_ids']]
    if any(np.any(np.linalg.norm(np.diff(x, axis=0), axis=1) < 1e-10) for x in (aa, bb)):
        return False
    # Test each strip in each face's own chart. Projecting an entire curved
    # face into another chart can wrap/fold even when the local join is sound.
    for patch,own,other in ((a,aa,bb),(b,bb,aa)):
        for k in range(len(own)-1):
            quad_xyz=np.array([own[k],own[k+1],other[k+1],other[k]])
            quad=curve_in_patch(patch,quad_xyz)
            strip=Polygon(quad)
            v=quad[1:]-quad[0]
            unsigned_area=sum(abs(v[i,0]*v[i+1,1]-v[i,1]*v[i+1,0])/2 for i in (0,1))
            if unsigned_area<=1e-12:
                continue
            if not strip.is_valid:
                return False
            far=primitive_residual(patch,other[k:k+2])
            separated=patch.kind=='plane' and (np.all(far>1e-7) or np.all(far < -1e-7))
            if not separated and strip.intersection(geometries[patch.id]).area>max(1e-12,strip.area*1e-7):
                return False
    return True


def _join_coplanar(patches, shared, candidates, cfg, diagnostics, log, existing_bridges=None, corner_gap=False):
    import copy
    existing_bridges = [] if existing_bridges is None else existing_bridges
    if (not corner_gap and not cfg.join_coplanar) or (corner_gap and not cfg.bridge_remaining_gaps):
        return shared, existing_bridges
    proposals = []
    stage = 'Corner-gap strips' if corner_gap else 'Joint coplanar borders'
    log(f'{stage}: searching {len(candidates)} candidate pairs')
    border_cache={}
    for i, j in candidates:
        pair_started=perf_counter()
        found=_coplanar_curves(patches[i], patches[j], cfg, corner_gap=corner_gap,border_cache=border_cache)
        proposals.extend(found)
        duration=perf_counter()-pair_started
        if duration>=.5:
            log(f'{stage}: pair {i}/{j}: {len(found)} curves in {duration:.3f}s')
    log(f'{stage}: validating {len(proposals)} proposed border curves')
    incident = {i: [e for e in shared+existing_bridges if i in e['patch_ids']] for i in patches}
    accepted, bridges = [], []
    last_report = perf_counter()
    for proposal_index, proposed in enumerate(sorted(proposals, key=lambda e: -np.linalg.norm(np.diff(e['points'], axis=0), axis=1).sum())):
        if proposal_index % 32 == 0 or perf_counter()-last_report >= 1:
            log(f'{stage}: validating {proposal_index+1}/{len(proposals)}; accepted {len(accepted)+len(bridges)}')
            last_report = perf_counter()
        edge = dict(proposed)
        if cfg.coplanar_join_mode == 'bridge' or corner_gap:
            edge['kind'] = 'bridge'
            edge['owner_id'] = min(edge['patch_ids'], key=lambda i: (patches[i].source_point_count, i))
        attempts = [edge]
        a, b = [patches[i] for i in edge['patch_ids']]
        angle = np.rad2deg(np.arccos(np.clip(abs(a.normal@b.normal),0,1)))
        separation = max(np.abs(primitive_residual(a,edge['side_points'][b.id])).max(),
                         np.abs(primitive_residual(b,edge['side_points'][a.id])).max())
        if a.kind == b.kind == 'plane' and edge.get('kind') == 'bridge' and not corner_gap and cfg.regularize_coplanar_borders and (angle > .1 or separation > 1e-5):
            common = edge['points']
            sides = {p.id:p.lift(p.project(common)) for p in (a,b)}
            # Node a crossing where a bevel shrinks to zero width instead of
            # drawing a bow tie through the intersection of the fitted planes.
            refined = {a.id:[],b.id:[]}
            for k in range(len(common)-1):
                for p in (a,b):refined[p.id].append(sides[p.id][k])
                d0=sides[a.id][k]-sides[b.id][k];d1=sides[a.id][k+1]-sides[b.id][k+1];delta=d1-d0
                t=-float(d0@delta)/max(float(delta@delta),1e-20)
                if 1e-6<t<1-1e-6 and np.linalg.norm(d0+t*delta)<1e-7:
                    crossing=(sides[a.id][k]+t*(sides[a.id][k+1]-sides[a.id][k])+sides[b.id][k]+t*(sides[b.id][k+1]-sides[b.id][k]))/2
                    for p in (a,b):refined[p.id].append(crossing)
            for p in (a,b):refined[p.id].append(sides[p.id][-1]);refined[p.id]=np.asarray(refined[p.id])
            # The setting requests a common simplified border on both planes.
            # Previously this was only a fallback after accepting the noisy
            # observed sides, so a valid jagged strip bypassed regularization.
            attempts.insert(0,dict(edge,side_points=refined,border_regularized=True))
        edits = []
        for trial in attempts:
            edits = []
            for label in trial['patch_ids']:
                patch = patches[label]
                global_edges = incident[label]+[trial]
                local_edges = [_local_edge(e, label) for e in global_edges]
                if not _consistent_vertex_targets(patch, local_edges, cfg.weld_tolerance):
                    break
                # Apply only the new edit to the current accepted boundary.
                # Replaying every previous chain from the alpha footprint for
                # every fragment makes heavily segmented borders quadratic.
                current = copy.copy(patch)
                current.footprint = patch.polygon
                geometry = _batch_boundary(current, [_local_edge(trial,label)], cfg)
                if geometry is None or not _preserves_local_locks(patch,geometry,
                        curve_in_patch(patch,_edge_points(_local_edge(trial,label))),
                        max(cfg.edge_tolerance,cfg.coplanar_boundary_tolerance)):
                    break
                edits.append((patch, geometry, global_edges, local_edges))
            if len(edits) == 2 and trial.get('kind') == 'bridge':
                if not _bridge_is_valid(trial, patches, {p.id: g for p, g, _, _ in edits}, cfg):
                    edits = []
            if len(edits) == 2:
                edge = trial
                break
        if len(edits) != 2:
            diagnostics.append(f"Pair {edge['patch_ids']}: rejected coplanar {cfg.coplanar_join_mode} (boundary conflict, crossing or overlapping strip)")
            continue
        for patch, geometry, global_edges, local_edges in edits:
            patch.polygon = geometry
            incident[patch.id] = global_edges
            patch.locked = [curve_in_patch(patch, curve)[k:k+2] for e in local_edges
                            for curve in [_edge_points(e)] for k in range(len(curve)-1)]
        (bridges if edge.get('kind') == 'bridge' else accepted).append(edge)
    stage = 'Corner-gap strips' if corner_gap else 'Joint coplanar borders'
    message = (f'{stage} ({cfg.coplanar_join_mode}): accepted '
               f'{len(accepted)+len(bridges)}/{len(proposals)} curves, patch IDs preserved')
    diagnostics.append(message)
    log(message)
    return shared+accepted, existing_bridges+bridges


def _simplify_candidate(patch, tolerance):
    """Douglas–Peucker on chains split at locked shared-edge endpoints."""
    polys = []
    locked = np.concatenate(patch.locked) if patch.locked else np.empty((0, 2))
    for poly in polygons(patch.polygon):
        rings = []
        for ring in [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]:
            anchors = [i for i, p in enumerate(ring) if len(locked) and np.min(np.linalg.norm(locked-p, axis=1)) < 1e-8]
            if not anchors:
                # Topology-preserving simplification for a closed ring.
                simplified = Polygon(ring).simplify(tolerance, preserve_topology=True)
                rings.append(np.asarray(simplified.exterior.coords)[:-1])
                continue
            if len(anchors) == 1:
                anchors.append((anchors[0]+len(ring)//2) % len(ring))
            anchors.sort()
            chunks = []
            for a, b in zip(anchors, anchors[1:]+anchors[:1]):
                chain = ring[(np.arange((b-a) % len(ring)+1)+a) % len(ring)]
                chunks.append(np.asarray(LineString(chain).simplify(tolerance).coords)[:-1])
            rings.append(np.concatenate(chunks))
        candidate_poly = Polygon(rings[0], rings[1:])
        if candidate_poly.is_empty or not candidate_poly.is_valid or candidate_poly.area <= 0:
            return None
        polys.append(candidate_poly)
    candidate = polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)
    if candidate.is_valid and _has_curves(candidate, patch.locked):
        return candidate
    return None


def _simplify(patch, tolerance, log=None):
    """Reject invalid ring edits before overlay; retry with smaller tolerances.

    Ring-by-ring Douglas–Peucker can move holes across shells or make separate
    components overlap. Validate the entire polygon collection without unioning
    (which could merge components). Preserve the original on any unsafe edit.
    """
    if tolerance == 0:
        return patch.polygon
    for factor in (1., .5, .25, .125):
        try:
            candidate = _simplify_candidate(patch, tolerance*factor)
        except (GEOSException, ValueError):
            candidate = None
        if candidate is not None:
            if factor != 1. and log is not None:
                log(f'Patch {patch.id}: unsafe simplification rejected; used tolerance {tolerance*factor:g}')
            return candidate
    if log is not None:
        log(f'Patch {patch.id}: unsafe simplification rejected; kept original boundary')
    return patch.polygon


def _close_junction_loops(vertices, faces, labels, parts, flags, patches, tolerance):
    """Cap only tiny multi-patch loops containing an existing bridge edge.

    Single-face holes, open chains and large outer boundaries remain untouched.
    Every filled loop is already a closed mesh-boundary cycle, so no nearest-point
    welding or arbitrary connecting diagonal can cross another surface boundary.
    """
    if tolerance <= 0 or not faces:
        return 0
    triangles = np.asarray(faces, dtype=int)
    edges = np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    unique, first, counts = np.unique(edges,axis=0,return_index=True,return_counts=True)
    edge_counts={tuple(edge):int(count) for edge,count in zip(unique,counts)}
    source = np.tile(np.arange(len(faces)),3)[first]
    boundary = unique[counts==1];source = source[counts==1]
    neighbors = {};ownership = {}
    for (a,b),face in zip(boundary,source):
        neighbors.setdefault(int(a),[]).append(int(b));neighbors.setdefault(int(b),[]).append(int(a))
        ownership[tuple(sorted((int(a),int(b))))] = int(face)
    seen = set();count = 0;xyz = np.asarray(vertices)
    for start in neighbors:
        if start in seen or len(neighbors[start])!=2:
            continue
        chain=[];previous=None;current=start
        while current not in chain and len(chain)<=128 and len(neighbors[current])==2:
            chain.append(current);seen.add(current)
            nxt = next(x for x in neighbors[current] if x!=previous)
            previous,current=current,nxt
        if current!=start or len(chain)<3:
            continue
        points=xyz[chain]
        if np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2))>tolerance:
            continue
        adjacent=[ownership[tuple(sorted((a,b)))] for a,b in zip(chain,chain[1:]+chain[:1])]
        patch_ids=set(int(labels[i]) for i in adjacent)
        if len(patch_ids)<2 or not any(flags[i] for i in adjacent):
            continue
        center=points.mean(axis=0);_,axes=np.linalg.eigh((points-center).T@(points-center));uv=(points-center)@axes[:,1:]
        polygon=Polygon(uv)
        if not polygon.is_valid or polygon.area<1e-12:
            continue
        caps=[]
        for triangle in shapely.constrained_delaunay_triangles(polygon).geoms:
            indices=cKDTree(uv).query(np.asarray(triangle.exterior.coords)[:3])[1]
            caps.append([chain[int(i)] for i in indices])
        from collections import Counter
        additions=Counter(tuple(sorted((triangle[i],triangle[j])))
                          for triangle in caps for i,j in ((0,1),(1,2),(2,0)))
        # A cap diagonal can coincide with an existing interior mesh edge,
        # particularly around thin corners. Do not create a third incident face.
        if any(edge_counts.get(edge,0)+count>2 for edge,count in additions.items()):
            continue
        owner=patches[min(patch_ids,key=lambda i:(patches[i].source_point_count,i))]
        for triangle in caps:
            faces.append(triangle);labels.append(owner.id);parts.append(-1 if owner.part_id is None else owner.part_id);flags.append(True)
        for edge,added in additions.items():edge_counts[edge]=edge_counts.get(edge,0)+added
        count+=1
    return count


def reconstruct(points, patch_ids, *, part_ids=None, mask=None, include_ids=None, config=None, verbose=False, progress=None):
    """Reconstruct planar patches. Inputs are aligned arrays, IDs are integers.

    Config.join_coplanar enables matching/simplifying opposing near-coplanar
    borders jointly. Plane offset and border separation have separate tolerances.
    Bridge mode preserves fitted planes and labels strip triangles with the smaller
    source patch ID (point count before subsampling; ties use the lower patch ID).
    Snap mode can move seam vertices off-plane. Result.bridges stores strip metadata;
    face_is_bridge identifies transition triangles and PLY exports is_bridge.
    Patch IDs and optional part-ID adjacency restrictions remain intact.
    Config.min_points filters retained original points BEFORE subsampling.
    Config.subsample_distance optionally sets minimum per-patch point spacing.
    PCA, alpha footprints, adjacency and edge support use the sampled points.
    mask filters individual points; include_ids filters patch IDs. Every retained
    patch must have one part ID. Invalid/nonplanar patches are reported and skipped.
    All original points/labels remain unmodified.
    verbose=True prints elapsed time, stages, patch progress, and pair decisions.
    Output is flushed immediately for notebooks and terminals.
    """
    started = perf_counter()
    def log(message):
        if progress is not None:
            progress(message)
        if verbose:
            print(f"[reconstruct +{perf_counter() - started:.1f}s] {message}", flush=True)
    log("Validating inputs and filtering points")
    cfg = config or Config()
    if not hasattr(shapely, 'constrained_delaunay_triangles'):
        raise RuntimeError('Requires Shapely >=2.1 and GEOS >=3.10')
    points = np.asarray(points, dtype=float)
    labels = np.asarray(patch_ids)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError('Expected points (N,3) and patch_ids (N,)')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('patch_ids must be integer labels')
    parts = None if part_ids is None else np.asarray(part_ids)
    if parts is not None and (parts.shape != labels.shape or not np.issubdtype(parts.dtype, np.integer)):
        raise ValueError('part_ids must be aligned integer labels')
    selected = np.isfinite(points).all(axis=1) & ~np.isin(labels, cfg.exclude_ids)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != labels.shape:
            raise ValueError('mask must have shape (N,)')
        selected &= m
    if include_ids is not None:
        selected &= np.isin(labels, list(include_ids))
    patches, diagnostics, shared = {}, [], []
    # Count once, discard small patches with a boolean mask, and group the
    # remaining point indices once rather than scanning N labels per patch.
    selected_ids = np.flatnonzero(selected)
    all_labels, inverse, counts = np.unique(
        labels[selected_ids], return_inverse=True, return_counts=True,
    )
    eligible = counts >= cfg.min_points
    point_mask = eligible[inverse]
    candidate_ids = selected_ids[point_mask]
    candidate_groups = inverse[point_mask]
    order = np.argsort(candidate_groups, kind='stable')
    grouped_ids = candidate_ids[order]
    retained_labels = all_labels[eligible]
    retained_counts = counts[eligible]
    offsets = np.concatenate(([0], np.cumsum(retained_counts)))
    removed_patches = int(np.count_nonzero(~eligible))
    removed_points = int(counts[~eligible].sum())
    if removed_patches:
        message = (f'Size filter: removed {removed_patches:,} patches / '
                   f'{removed_points:,} points below min_points={cfg.min_points}')
        diagnostics.append(message)
        log(message)
    log(f"Retained {len(grouped_ids):,}/{len(points):,} points in {len(retained_labels)} patches")
    for number, label in enumerate(retained_labels, 1):
        log(f"PCA/alpha {number}/{len(retained_labels)}: patch {label}")
        ids = grouped_ids[offsets[number-1]:offsets[number]]
        part = None
        if parts is not None:
            unique = np.unique(parts[ids])
            if len(unique) != 1:
                raise ValueError(f'Patch {label} spans multiple part IDs; use globally unique patch IDs')
            part = int(unique[0])
        xyz = points[ids]
        if cfg.subsample_distance is not None:
            log(f"Patch {label}: subsampling {len(xyz):,} points at {cfg.subsample_distance:g} spacing")
            xyz = spatial_subsample(xyz, cfg.subsample_distance)
            log(f"Patch {label}: retained {len(xyz):,}/{len(ids):,} sampled points")
            if len(xyz) < 3:
                diagnostics.append(f'Patch {label}: skipped (fewer than 3 points after subsampling)')
                log(diagnostics[-1])
                continue
        center = xyz.mean(axis=0)
        values, vectors = np.linalg.eigh((xyz-center).T @ (xyz-center)/len(xyz))
        rmse = np.sqrt(max(0, values[0]))
        plane_rmse = rmse
        cylinder = None
        spline = None
        basis = vectors[:, 1:]
        normal = np.cross(basis[:, 0], basis[:, 1])
        if cfg.fit_cylinders and rmse > cfg.cylinder_plane_trigger:
            log(f'Patch {label}: comparing plane and cylinder fits')
            fitted = fit_cylinder(xyz, max_radius=cfg.cylinder_max_radius)
            if fitted is not None and fitted.rmse < min(cfg.max_cylinder_rmse, plane_rmse*cfg.cylinder_error_ratio) and np.rad2deg(fitted.angular_span) >= cfg.cylinder_min_angle_degrees:
                cylinder = fitted
                rmse = fitted.rmse
        if cylinder is None and cfg.fit_bsplines and plane_rmse > cfg.max_plane_rmse:
            # On a long curved web the vertical thickness can have less variance
            # than the transverse bend. The least-variance PCA direction is then
            # the wrong height axis. Try the other PCA charts without changing
            # the plane/cylinder selection or the geometric error requirements.
            for axis, columns in enumerate(([1,2], [2,0], [0,1])):
                log(f'Patch {label}: fitting B-spline in PCA chart {axis}')
                chart = vectors[:, columns]
                spline, reason = fit_bspline(xyz, center, chart, np.cross(chart[:,0],chart[:,1]), cfg)
                if spline is not None and spline.rmse > cfg.bspline_error_ratio*plane_rmse:
                    spline, reason = None, f'spline does not improve best PCA plane RMSE={plane_rmse:.6g}'
                diagnostics.append(f'Patch {label}: PCA height axis {axis}: {reason}')
                log(diagnostics[-1])
                if spline is not None:
                    break
            if spline is not None:
                rmse = spline.rmse
        if cylinder is None and spline is None and (rmse > cfg.max_plane_rmse or values[1] <= max(values[2]*1e-10, 1e-16)):
            diagnostics.append(f'Patch {label}: skipped (nonplanar/degenerate, RMSE={rmse:.6g})')
            log(diagnostics[-1]);continue
        basis = vectors[:, 1:]
        normal = np.cross(basis[:, 0], basis[:, 1])
        if spline is not None:
            basis, normal = spline.basis, spline.normal
        patch = Patch(int(label), part, center, normal, basis, xyz, None, None, rmse,
                      cylinder=cylinder, spline=spline, plane_rmse=plane_rmse, source_point_count=len(ids))
        if cylinder is not None:
            message = (f'Patch {label}: cylinder r={cylinder.radius:.4f}, RMSE={rmse:.5f} '
                       f'(plane {plane_rmse:.5f}), coverage={np.rad2deg(cylinder.angular_span):.1f} degrees')
            diagnostics.append(message);log(message)
        elif spline is None:
            message = f'Patch {label}: plane, RMSE={plane_rmse:.6g}'
            diagnostics.append(message);log(message)
        try:
            uv = patch.project(xyz)
            if cylinder is not None and cylinder.periodic:
                from shapely.geometry import box
                period = 2*np.pi*cylinder.radius
                margin = cfg.alpha_radius*2
                if cfg.cylinder_trim_mode == 'extent':
                    footprint = box(0, uv[:,1].min(), period, uv[:,1].max())
                    diagnostics.append(f'Patch {label}: extent-trimmed complete cylinder; scan holes are not retained')
                else:
                    extended = np.vstack([uv, uv[uv[:,0]<margin]+[period,0], uv[uv[:,0]>period-margin]-[period,0]])
                    footprint = alpha_footprint(extended, cfg.alpha_radius, log=log)
                    footprint = footprint.intersection(box(0, uv[:,1].min()-margin, period, uv[:,1].max()+margin))
            else:
                footprint = alpha_footprint(uv, cfg.alpha_radius, log=log)
            patch.footprint = patch.polygon = footprint
        except (ValueError, QhullError) as exc:
            diagnostics.append(f'Patch {label}: skipped ({exc})');log(diagnostics[-1]);continue
        patches[int(label)] = patch
    log(f"Building adjacency graph for {len(patches)} fitted patches")
    candidate_pairs = list(_candidate_pairs(patches, cfg, log))
    if cfg.clean_first:
        from .clean_joints import reconstruct_joints
        shared, bridges = reconstruct_joints(patches, candidate_pairs, cfg, diagnostics, log)
    else:
        proposals = []
        for i, j in candidate_pairs:
            log(f'Pair {i}/{j}: computing and validating surface intersections')
            try:
                def pair_diagnostic(message):
                    diagnostics.append(f'Pair {i}/{j}: {message}')
                    log(diagnostics[-1])
                segments = _intersection_segments(patches[i], patches[j], cfg, pair_diagnostic)
            except RuntimeError as exc:
                diagnostics.append(f'Pair {(i,j)}: intersection failed: {exc}')
                log(diagnostics[-1])
                continue
            proposals.extend({'patch_ids': (i, j), 'points': xyz, 'endpoints': xyz[[0,-1]].copy()} for xyz in segments)
            diagnostics.append(f'Pair {i}/{j}: proposed {len(segments)} supported segments')
            log(diagnostics[-1])
        log(f'Collected {len(proposals)} shared-edge proposals; reconciling corners')
        _reconcile_corners(patches, proposals, cfg,
                           lambda message: (diagnostics.append(message), log(message)))
        shared = _assemble_topology(patches, proposals, cfg, diagnostics, log)
        shared = _rescue_edges(patches, proposals, shared, cfg, diagnostics, log)
        shared, bridges = _join_coplanar(patches, shared, candidate_pairs, cfg, diagnostics, log)
        # Joint border edits can expose a valid chain for a previously rejected pair.
        if cfg.rescue_shared_edges:
            import copy
            accepted_pairs = {frozenset(e['patch_ids']) for e in shared}
            extras = []
            for i, j in candidate_pairs:
                if frozenset((i,j)) in accepted_pairs:
                    continue
                aa, bb = copy.copy(patches[i]), copy.copy(patches[j])
                aa.footprint, bb.footprint = aa.polygon, bb.polygon
                try:
                    segments = _intersection_segments(aa,bb,cfg)
                except RuntimeError as exc:
                    diagnostics.append(f'Pair {(i,j)}: rescue intersection failed: {exc}')
                    log(diagnostics[-1])
                    continue
                extras.extend({'patch_ids':(i,j),'points':curve,'endpoints':curve[[0,-1]].copy()}
                              for curve in segments)
            shared = _rescue_edges(patches, extras, shared, cfg, diagnostics, log)
        shared, bridges = _join_coplanar(patches, shared, candidate_pairs, cfg, diagnostics, log,
                                        existing_bridges=bridges, corner_gap=True)
        if cfg.clean_border_pass:
            log('Clean-border pass: simplify exposed runs, then reconnect remaining supported borders')
            for patch in patches.values():
                patch.polygon = _simplify(patch, cfg.simplify_tolerance)
                patch.footprint = patch.polygon
            shared, bridges = _join_coplanar(patches, shared, candidate_pairs, cfg, diagnostics, log,
                                            existing_bridges=bridges)
            shared, bridges = _join_coplanar(patches, shared, candidate_pairs, cfg, diagnostics, log,
                                            existing_bridges=bridges, corner_gap=True)
    from .meshing import (spline_grid_spacing, cylinder_grid_spacing, synchronize_seam_grid,
                          insert_boundary_nodes, grid_triangles)
    _synchronize_existing_seam_nodes(shared+bridges,patches)
    spacings={label:spline_grid_spacing(p.spline,cfg.bspline_mesh_spacing,cfg.curve_tolerance)
              for label,p in patches.items() if p.kind=='bspline'}
    spacings.update({label:cylinder_grid_spacing(p.cylinder,cfg.cylinder_mesh_spacing,cfg.cylinder_chord_tolerance)
                     for label,p in patches.items() if p.kind=='cylinder'})
    if spacings:
        log('Synchronizing shared seams with curved-surface UV grids')
        for edge in shared+bridges:
            synchronize_seam_grid(edge,patches,spacings)
    if shared or bridges:
        for label,patch in patches.items():
            incident=[_local_edge(e,label) for e in shared+bridges if label in e['patch_ids']]
            if incident:
                try:
                    patch.polygon=insert_boundary_nodes(patch.polygon,
                        [_mesh_edge_uv(patch,e) for e in incident])
                except ValueError as exc:
                    raise ValueError(f'Patch {label}: {exc}') from exc
                patch.locked=[_mesh_edge_uv(patch,e)[k:k+2]
                              for e in incident for k in range(len(_edge_points(e))-1)]
    for edge in shared:
        edge['uv_curves'] = {label: _mesh_edge_uv(patches[label], edge)
                             for label in edge['patch_ids']}
    vertices, faces, face_ids, part_labels, bridge_flags = [], [], [], [], []
    # Spatial welding only: tolerance must be far below real feature separation.
    buckets = {}
    def vertex_index(xyz):
        cell = tuple(np.floor(xyz/cfg.weld_tolerance).astype(np.int64))
        from itertools import product
        for delta in product((-1, 0, 1), repeat=3):
            key = tuple(x+d for x, d in zip(cell, delta))
            for index in buckets.get(key, []):
                if np.linalg.norm(vertices[index]-xyz) <= cfg.weld_tolerance:
                    return index
        index = len(vertices)
        vertices.append(xyz)
        buckets.setdefault(cell, []).append(index)
        return index
    log("Triangulating coordinated clean boundaries" if cfg.clean_first else
        "Simplifying locked boundaries and constrained retriangulation")
    for number, (label, patch) in enumerate(patches.items(), 1):
        log(f"{'Triangulate' if cfg.clean_first else 'Simplify/triangulate'} {number}/{len(patches)}: patch {label}")
        if not cfg.clean_first:
            patch.polygon = _simplify(
                patch, cfg.simplify_tolerance,
                log=lambda message: (diagnostics.append(message), log(message)),
            )
        incident = [_local_edge(e, label) for e in shared+bridges if label in e['patch_ids']]
        boundary_xyz = np.concatenate([_edge_points(e) for e in incident]) if incident else None
        boundary_uv = np.concatenate([_mesh_edge_uv(patch,e) for e in incident]) if incident else None
        if incident and patch.cylinder is not None and patch.cylinder.periodic:
            period = 2*np.pi*patch.cylinder.radius
            boundary_uv = np.vstack([boundary_uv, boundary_uv+[period,0], boundary_uv-[period,0]])
            boundary_xyz = np.tile(boundary_xyz,(3,1))
        boundary_tree = cKDTree(boundary_uv) if incident else None
        if patch.kind == 'bspline':
            locked = [_mesh_edge_uv(patch,e) for e in incident]
            uv_triangles = refine_surface_triangles(list(polygons(patch.polygon)), patch.lift,
                                                   cfg.curve_tolerance, locked, spacing=spacings[label])
        elif patch.kind == 'plane':
            uv_triangles = [np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(patch.polygon).geoms]
        else:
            uv_triangles = grid_triangles(list(polygons(patch.polygon)),spacings[label])
        for uv in uv_triangles:
            xyz = patch.lift(uv)
            if boundary_tree is not None:
                distance, index = boundary_tree.query(uv)
                matched = distance < 1e-8
                # Nearly coplanar shared seams intentionally relax plane fidelity
                # only at border vertices so both meshes have identical 3D edges.
                xyz[matched] = boundary_xyz[index[matched]]
            expected_normal = patch.normal_at(np.mean(uv, axis=0))
            if np.dot(np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0]), expected_normal) < 0:
                xyz = xyz[[0, 2, 1]]
            indices = [vertex_index(p) for p in xyz]
            if len(set(indices)) < 3:
                diagnostics.append(f'Patch {label}: dropped triangle collapsed by weld tolerance')
                continue
            faces.append(indices)
            face_ids.append(label)
            part_labels.append(-1 if patch.part_id is None else patch.part_id)
            bridge_flags.append(False)
    bridge_triangles = 0
    for bridge in bridges:
        a, b = [bridge['side_points'][i] for i in bridge['patch_ids']]
        owner = patches[bridge['owner_id']]
        for k in range(len(a)-1):
            # Corresponding source-side curves share the same DP breakpoints.
            for triangle in ([a[k], a[k+1], b[k+1]], [a[k], b[k+1], b[k]]):
                xyz = np.asarray(triangle)
                normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
                if np.linalg.norm(normal) <= 1e-12:
                    continue
                if normal @ owner.normal_at(owner.project(xyz.mean(axis=0))) < 0:
                    xyz = xyz[[0, 2, 1]]
                indices = [vertex_index(point) for point in xyz]
                if len(set(indices)) < 3:
                    continue
                faces.append(indices)
                face_ids.append(owner.id)
                part_labels.append(-1 if owner.part_id is None else owner.part_id)
                bridge_flags.append(True)
                bridge_triangles += 1
    if cfg.stitch_mesh_boundaries:
        vertices, faces, face_ids, part_labels, bridge_flags, stitched, stitch_triangles = stitch_boundaries(
            vertices, faces, face_ids, part_labels, bridge_flags, patches,
            cfg.mesh_stitch_tolerance, cfg.min_edge_length, cfg.weld_tolerance, cfg=cfg)
        bridges.extend(stitched)
        bridge_triangles += stitch_triangles
        log(f'Mesh boundary stitching: {len(stitched)} overlap intervals, {stitch_triangles} strip triangles')
    caps = _close_junction_loops(vertices, faces, face_ids, part_labels, bridge_flags,
                                 patches, cfg.junction_cap_tolerance)
    log(f'Junction cleanup: capped {caps} tiny multi-patch bridge loops')
    log(f'Bridges: {len(bridges)} strips, {bridge_triangles} triangles owned by smaller source patch')
    log(f"Done: {len(patches)} patches, {len(shared)} shared edges, {len(vertices):,} vertices, {len(faces):,} triangles")
    return Result(np.asarray(vertices).reshape(-1, 3), np.asarray(faces, dtype=np.int32).reshape(-1, 3),
                  np.asarray(face_ids), np.asarray(part_labels), patches, shared, diagnostics,
                  bridges=bridges, face_is_bridge=np.asarray(bridge_flags, dtype=bool))


def reconstruct_ply(path, patch_field, *, part_field=None, **kwargs):
    from plyfile import PlyData
    data = PlyData.read(str(path))['vertex'].data
    xyz = np.column_stack([data[name] for name in ('x', 'y', 'z')])
    return reconstruct(xyz, data[patch_field], part_ids=None if part_field is None else data[part_field], **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--patch-field', required=True)
    parser.add_argument('--part-field')
    parser.add_argument('--verbose', action='store_true', help='Print elapsed time and reconstruction progress')
    parser.add_argument('--config', type=Path, help='JSON object matching Config fields')
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    result = reconstruct_ply(args.input, args.patch_field, part_field=args.part_field, config=cfg, verbose=args.verbose)
    result.export_ply(args.output)
    print(f'{len(result.patches)} patches, {len(result.shared_edges)} shared edges, {len(result.faces)} triangles')
    print('\n'.join(result.diagnostics))
