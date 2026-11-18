"""Local surface charts and regularized bicubic B-spline height fitting.

Chart projection is deliberately not closest-point projection. The former is
used for footprints and field transfer; OCCT supplies geometric fit distances.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import BSpline


def _clamped_knots(count):
    return np.r_[np.zeros(4), np.linspace(0., 1., count-2)[1:-1], np.ones(4)]


@dataclass
class Plane:
    center: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    kind = "plane"

    def project(self, xyz):
        return (np.asarray(xyz) - self.center) @ self.basis

    def lift(self, uv):
        return self.center + np.asarray(uv) @ self.basis.T

    def normal_at(self, uv):
        return np.broadcast_to(self.normal, np.asarray(uv).shape[:-1] + (3,))

    def residual(self, xyz):
        return (np.asarray(xyz) - self.center) @ self.normal

    def descriptor(self, bounds):
        return dict(kind=self.kind, origin=self.center, normal=self.normal,
                    u=self.basis[:, 0], bounds=bounds)


def _bernstein(t):
    t = np.asarray(t)
    return np.stack(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3), axis=-1)


def _derivative(t):
    t = np.asarray(t)
    return np.stack((-3*(1-t)**2, 3*(1-t)*(1-3*t), 3*t*(2-3*t), 3*t*t), axis=-1)


@dataclass
class BSplineHeight:
    center: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    bounds: np.ndarray  # Physical local chart coordinates, including bounded halo.
    heights: np.ndarray  # Non-rational, clamped cubic control heights.
    rmse: float = 0.
    knots_u: np.ndarray | None = None
    knots_v: np.ndarray | None = None
    kind = "bspline"

    def __post_init__(self):
        if self.knots_u is None:
            self.knots_u = _clamped_knots(self.heights.shape[0])
        if self.knots_v is None:
            self.knots_v = _clamped_knots(self.heights.shape[1])

    @cached_property
    def _u_basis(self):
        return BSpline(self.knots_u, np.eye(self.heights.shape[0]), 3)

    @cached_property
    def _v_basis(self):
        return BSpline(self.knots_v, np.eye(self.heights.shape[1]), 3)

    def project(self, xyz):
        return (np.asarray(xyz) - self.center) @ self.basis

    def _parameters(self, uv):
        return (np.asarray(uv) - self.bounds[:2]) / (self.bounds[2:] - self.bounds[:2])

    def lift(self, uv):
        uv = np.asarray(uv)
        t = self._parameters(uv)
        h = np.einsum('...i,ij,...j->...', self._u_basis(t[..., 0]), self.heights, self._v_basis(t[..., 1]))
        return self.center + uv @ self.basis.T + h[..., None] * self.normal

    def gradient(self, uv):
        t = self._parameters(uv)
        span = self.bounds[2:] - self.bounds[:2]
        du = np.einsum('...i,ij,...j->...', self._u_basis(t[..., 0], nu=1), self.heights, self._v_basis(t[..., 1])) / span[0]
        dv = np.einsum('...i,ij,...j->...', self._u_basis(t[..., 0]), self.heights, self._v_basis(t[..., 1], nu=1)) / span[1]
        return np.stack((du, dv), axis=-1)

    def normal_at(self, uv):
        n = self.normal - self.gradient(uv) @ self.basis.T
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def residual(self, xyz):
        xyz = np.asarray(xyz)
        # Signed distance requires a true closest-surface projection, not chart height.
        from .geometry_backend import closest_points
        flat = xyz.reshape(-1, 3)
        closest = closest_points(self, flat)
        sign = np.where(np.einsum('ij,ij->i', flat-closest, self.normal_at(self.project(closest))) >= 0, 1., -1.)
        return (np.linalg.norm(flat-closest, axis=1) * sign).reshape(xyz.shape[:-1])

    def descriptor(self, bounds=None):
        # Greville abscissae reproduce the linear chart exactly for multiple
        # spans. Uniformly spaced poles only work for the single-span 4x4 case.
        u = self.bounds[0] + (self.bounds[2]-self.bounds[0])*np.array([
            self.knots_u[i+1:i+4].mean() for i in range(self.heights.shape[0])])
        v = self.bounds[1] + (self.bounds[3]-self.bounds[1])*np.array([
            self.knots_v[i+1:i+4].mean() for i in range(self.heights.shape[1])])
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)
        poles = self.center + uv @ self.basis.T + self.heights[..., None] * self.normal
        if bounds is None:
            domain = [0., 0., 1., 1.]
        else:
            corners = self._parameters(np.asarray(bounds).reshape(2, 2))
            domain = np.clip(corners.reshape(-1), 0., 1.).tolist()
        return dict(kind=self.kind, poles=poles, bounds=domain,
                    u_knots=self.knots_u, v_knots=self.knots_v)


def fit_bspline(points, center, basis, normal, cfg):
    """Fit the smallest supported cubic height patch; return (surface, diagnostic).

    Start at 4x4 poles. Longer surfaces can need several cubic spans even with
    low curvature. Refine up to 12 poles along the long direction (up to 12x8
    for balanced charts), retaining the same geometric holdout requirements.
    """
    if len(points) < 64:
        return None, 'fewer than 64 points for spline validation'
    uv = (np.asarray(points)-center) @ basis
    span = np.ptp(uv, axis=0)
    grids = [(4,4)]
    if span.max() > 0 and span.min() > 0:
        long = int(np.argmax(span))
        for size in (6,8,12):
            short = 4 if span.max()/span.min() >= 3 else min(size,8)
            grids.append((size,short) if long == 0 else (short,size))
    failures = []
    for shape in grids:
        surface, reason = _fit_bspline_grid(points, center, basis, normal, cfg, shape)
        if surface is not None:
            return surface, reason
        failures.append(f'{shape[0]}x{shape[1]}: {reason}')
        if reason in {'fewer than 64 points for spline validation', 'degenerate spline chart',
                      'insufficient two-dimensional spline support', 'multilayer/noisy spline support'}:
            break
    return None, '; '.join(failures)


def _fit_bspline_grid(points, center, basis, normal, cfg, shape):
    """Fit one control grid with spatial validation and robust regularization.

    An entire checkerboard of UV cells is held out, rather than randomly holding
    out points next to training samples. Acceptance uses OCCT geometric distances.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 64:
        return None, "fewer than 64 points for spline validation"
    uv = (points-center) @ basis
    heights = (points-center) @ normal
    low, high = uv.min(axis=0), uv.max(axis=0)
    span = high-low
    scale = float(span.max())
    if scale <= 0 or span.min() <= scale*1e-6:
        return None, "degenerate spline chart"
    bounds = np.r_[low-cfg.edge_tolerance, high+cfg.edge_tolerance]
    t = (uv-bounds[:2])/(bounds[2:]-bounds[:2])
    nu, nv = shape
    ku, kv = _clamped_knots(nu), _clamped_knots(nv)
    cells = np.minimum(4, ((uv-low)/span*5).astype(int))
    validation = (cells[:, 0]+2*cells[:, 1]) % 5 == 0
    if validation.sum() < 10 or (~validation).sum() < max(32, 2*nu*nv) or len(np.unique(cells, axis=0)) < 16:
        return None, "insufficient two-dimensional spline support"
    # Stratify both spatial coverage and the existing holdout partition before
    # constructing the tall design matrix. Repeated dense scan samples must not
    # make each robust least-squares iteration grow with the full cloud size.
    # Multiplicities retain the original density-weighted objective.
    dense_cells = np.minimum(7, ((uv-low)/span*8).astype(int))
    dense_ids = dense_cells[:, 0]*8+dense_cells[:, 1]
    sample, multiplicity = _fit_sample(dense_ids*2+validation.astype(int))
    st = t[sample]
    design = np.einsum('ni,nj->nij', BSpline(ku,np.eye(nu),3)(st[:, 0]),
                      BSpline(kv,np.eye(nv),3)(st[:, 1])).reshape(-1, nu*nv)
    sampled_validation = validation[sample]
    singular = np.linalg.svd(design[~sampled_validation], compute_uv=False)
    if singular[-1] < singular[0]*1e-8:
        return None, "rank-deficient spline fit"
    # Bending penalty in scale-normalized physical coordinates (not world XY).
    # Control derivatives use knot spacing, including the shorter end spans.
    def derivative_operators(knots, count):
        first = 3*np.diff(np.eye(count),axis=0)/(knots[4:count+3]-knots[1:count])[:,None]
        second = 2*np.diff(first,axis=0)/(knots[4:count+2]-knots[2:count])[:,None]
        return first, second
    u1,u2 = derivative_operators(ku,nu)
    v1,v2 = derivative_operators(kv,nv)
    widths = (bounds[2:]-bounds[:2])/scale
    penalty = np.vstack((np.kron(u2, np.eye(nv))/widths[0]**2,
                         np.sqrt(2)*np.kron(u1, v1)/(widths[0]*widths[1]),
                         np.kron(np.eye(nu), v2)/widths[1]**2))
    # Normalize the operator, not just coordinates: the user-facing smoothing
    # weight should not scale with the number of derivative constraints.
    penalty /= np.linalg.norm(penalty, ord=2)
    density = np.bincount(dense_ids, minlength=64)
    base_weights = multiplicity / density[dense_ids[sample]]

    def solve(mask):
        x, y = design[mask], heights[sample[mask]]/scale
        weights = base_weights[mask]
        weights = weights / weights.mean()
        robust = np.ones(len(y))
        coef = np.zeros(nu*nv)
        for _ in range(6):
            w = np.sqrt(weights*robust/len(y))
            coef = np.linalg.lstsq(np.vstack((x*w[:, None], np.sqrt(cfg.bspline_smoothing)*penalty)),
                                  np.r_[y*w, np.zeros(len(penalty))], rcond=None)[0]
            error = x@coef-y
            sigma = max(1e-9, 1.4826*np.median(np.abs(error-np.median(error))))
            robust = np.minimum(1., 1.345*sigma/np.maximum(np.abs(error), 1e-15))
        return BSplineHeight(center, normal, basis, bounds, coef.reshape(nu,nv)*scale)

    def slope_ok(surface):
        # Derivative poles bound the gradient throughout the entire chart.
        du = np.abs(u1@surface.heights).max()/(bounds[2]-bounds[0])
        dv = np.abs(surface.heights@v1.T).max()/(bounds[3]-bounds[1])
        return np.hypot(du, dv) <= cfg.bspline_max_slope

    from .geometry_backend import closest_points
    surface = solve(~sampled_validation)
    if not slope_ok(surface):
        return None, "spline slope exceeds limit"
    held = np.flatnonzero(validation)
    # Deterministic spatial validation is bounded for large scan segments.
    if len(held) > 2048:
        held = held[np.linspace(0, len(held)-1, 2048).astype(int)]
    try:
        error = np.linalg.norm(points[held]-closest_points(surface, points[held]), axis=1)
    except RuntimeError as exc:
        return None, f"spline projection failed: {exc}"
    rmse = float(np.sqrt(np.mean(error**2)))
    plane_rmse = float(np.sqrt(np.mean(heights[held]**2)))
    if rmse > cfg.max_bspline_rmse or rmse > cfg.bspline_error_ratio*plane_rmse:
        return None, f"spline holdout RMSE {rmse:.6g} fails acceptance (plane {plane_rmse:.6g})"
    surface = solve(np.ones(len(sample), dtype=bool))
    if not slope_ok(surface):
        return None, "refitted spline slope exceeds limit"
    # Check all input points in chart-height distance: conservative for low slopes,
    # and prevents a well-fit subset from concealing a second surface layer.
    squared_error = 0.
    for start in range(0, len(points), 65536):
        error = ((points[start:start+65536]-surface.lift(uv[start:start+65536])) @ normal)
        squared_error += float(error @ error)
    if np.sqrt(squared_error/len(points)) > cfg.max_bspline_rmse*np.sqrt(1+cfg.bspline_max_slope**2):
        return None, "multilayer/noisy spline support"
    surface.rmse = rmse
    return surface, f"bicubic B-spline: holdout RMSE={rmse:.6g}, chart-plane={plane_rmse:.6g}, {nu}x{nv} poles"


def _fit_sample(strata, per_stratum=128):
    """Bound training rows, retaining each occupied spatial/holdout stratum.

    At most 128 strata times 128 points (16,384 rows). Stable input-order
    quantiles avoid first-point bias; weights represent the full stratum count.
    Strata with at most 128 points retain every point and its original weight.
    """
    order = np.argsort(strata, kind='stable')
    _, starts, counts = np.unique(strata[order], return_index=True, return_counts=True)
    rows, weights = [], []
    for start, count in zip(starts, counts):
        take = min(int(count), per_stratum)
        offsets = np.linspace(0, count-1, take).astype(np.intp)
        rows.append(order[start+offsets])
        weights.append(np.full(take, count/take))
    rows, weights = np.concatenate(rows), np.concatenate(weights)
    sorting = np.argsort(rows)
    return rows[sorting], weights[sorting]
