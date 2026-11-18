"""One OCCT backend for all surface-pair combinations; chart trimming stays in Python."""
import numpy as np

from .surfaces import Plane


def _native():
    try:
        from geon._native import surface_geometry
    except ImportError as exc:
        raise RuntimeError("Reconstruction needs geon's OCCT surface_geometry extension. Rebuild/install geon with submodules initialized.") from exc
    return surface_geometry


def surface_of(patch):
    if hasattr(patch, "surface"):
        return patch.surface
    if getattr(patch, "cylinder", None) is not None:
        return patch.cylinder
    normal = np.asarray(patch.normal)
    basis = getattr(patch, "basis", None)
    if basis is None:
        u = np.eye(3)[np.argmin(np.abs(normal))]
        u = u-normal*(u@normal)
        u /= np.linalg.norm(u)
        basis = np.column_stack((u, np.cross(normal, u)))
    return Plane(patch.center, normal, basis)


def patch_surface(patch, margin=0., reference=None):
    model = surface_of(patch)
    footprint = getattr(patch, "footprint", None)
    if hasattr(patch, 'parameter_bounds'):
        bounds = np.asarray(patch.parameter_bounds, dtype=float).copy()
    elif footprint is not None:
        bounds = np.asarray(footprint.bounds, dtype=float)
    elif hasattr(patch, "points"):
        uv = model.project(patch.points)
        bounds = np.r_[uv.min(axis=0), uv.max(axis=0)]
    elif reference is not None:
        other = surface_of(reference)
        b = reference.footprint.bounds
        uv = model.project(other.lift(np.array([[b[0],b[1]], [b[0],b[3]], [b[2],b[1]], [b[2],b[3]]])))
        bounds = np.r_[uv.min(axis=0)-1, uv.max(axis=0)+1]
    else:
        raise ValueError("Surface patch requires finite bounds")
    bounds += np.array([-margin, -margin, margin, margin])
    bounds[2:] = np.maximum(bounds[2:], bounds[:2]+1e-8)
    if model.kind == "bspline":
        bounds[:2] = np.maximum(bounds[:2], model.bounds[:2])
        bounds[2:] = np.minimum(bounds[2:], model.bounds[2:])
    return _native().Surface(model.descriptor(bounds))


def intersection_curves(a, b, step, chord_tolerance=.005, margin=0., diagnostic=None):
    native = _native()
    sa, sb = patch_surface(a, margin, b), patch_surface(b, margin, a)
    curves = native.intersections(sa, sb, step, 1e-7, chord_tolerance)
    report = diagnostic or (lambda message: None)
    if not curves:
        # IntSS does not classify isolated tangencies or coincident domains.
        # State this ambiguity rather than turning it into a synthetic seam.
        report('OCCT returned no intersection curve (disjoint, isolated tangent, or coincident domains)')
    transverse = []
    am, bm = surface_of(a), surface_of(b)
    for curve in curves:
        if len(curve) < 2 or np.ptp(curve, axis=0).max() < 1e-8:
            report('isolated tangency: no seam')
            continue
        an, bn = am.normal_at(am.project(curve)), bm.normal_at(bm.project(curve))
        if np.all(np.linalg.norm(np.cross(an, bn), axis=1) <= 1e-6):
            report('tangent intersection / ambiguous overlap: no seam')
            continue
        transverse.append(curve)
    return _join_curve_branches(transverse)


def _join_curve_branches(curves, tolerance=1e-6):
    """Reconnect OCCT periodic-seam splits by tangent continuity, not proximity alone."""
    curves = [np.asarray(c) for c in curves if len(c) >= 2]
    while True:
        best = None
        for i, a in enumerate(curves):
            if np.linalg.norm(a[0]-a[-1]) <= tolerance:
                continue
            for j in range(i+1, len(curves)):
                b = curves[j]
                if np.linalg.norm(b[0]-b[-1]) <= tolerance:
                    continue
                for ai in (0, -1):
                    for bi in (0, -1):
                        if np.linalg.norm(a[ai]-b[bi]) > tolerance:
                            continue
                        da = a[1 if ai == 0 else -2]-a[ai]
                        db = b[1 if bi == 0 else -2]-b[bi]
                        cosine = float(da@db/max(np.linalg.norm(da)*np.linalg.norm(db), 1e-30))
                        if cosine < -.9 and (best is None or cosine < best[0]):
                            best = cosine, i, j, ai, bi
        if best is None:
            return curves
        _, i, j, ai, bi = best
        a = curves[i][::-1] if ai == 0 else curves[i]
        b = curves[j][::-1] if bi == -1 else curves[j]
        curves[i] = np.vstack((a, b[1:]))
        del curves[j]


def closest_points(model, points):
    native = _native()
    surface = native.Surface(model.descriptor())
    return native.closest_points(surface, np.ascontiguousarray(points, dtype=float))


def triple_points(patches, margin):
    surfaces = [patch_surface(p, margin) for p in patches]
    return _native().triple_points(*surfaces)
