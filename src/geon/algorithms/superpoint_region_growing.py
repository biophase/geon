"""Experimental planar growth over indivisible instance-labelled superpoints.

Only NumPy/SciPy are required. Distances use coordinate units, angles use degrees,
``tau`` counts original points, and ``first_refit`` counts superpoints. Negative
input labels are ignored. Selection expands to complete source instances.

Example::

    labels, stats = segment_planar_superpoints(
        cloud, instance_field_name="superpoints",
        params={"min_d": 0.05, "epsilon": 0.03, "tau": 80},
    )

``progress_callback(stage, done, total)`` runs on the caller's thread.
``cancel_requested()`` is polled between work units and raises
``SuperpointGrowthCancelled``; no partial labels are returned. This prototype
uses deterministic seeds and core planar growth, not native reconciliation,
reassignment, or probabilistic stopping.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from .touching_graph import build_touching_graph


DEFAULT_PARAMS = {
    "min_d": 0.05,
    "epsilon": 0.03,
    "epsilon_multiplier": 3.0,
    "alpha_deg": 29.0,
    "tau": 80,
    "first_refit": 4,
    "refit_multiplier": 2.0,
    "max_dist_from_cent": 50.0,
    "enable_seed_gating": True,
    "seed_min_points": 10,
    "seed_planarity_min": 0.20,
    "seed_scattering_max": 0.35,
}


class SuperpointGrowthCancelled(RuntimeError):
    """The caller cancelled the computation; discard this run."""


def _parameters(params: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_PARAMS)
    if params is not None:
        unknown = params.keys() - result.keys()
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
        result.update(params)
    for key, value in result.items():
        if key == "enable_seed_gating":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean")
        elif key in {"tau", "first_refit", "seed_min_points"}:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        else:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            lower = 0 if key in {"min_d", "seed_planarity_min", "seed_scattering_max"} else 0.0
            if value < lower or (value == 0 and key not in {"min_d", "seed_planarity_min", "seed_scattering_max"}):
                raise ValueError(f"{key} is outside its valid range")
            if key in {"seed_planarity_min", "seed_scattering_max"} and value > 1:
                raise ValueError(f"{key} must be between 0 and 1")
            if key == "alpha_deg" and value > 90:
                raise ValueError("alpha_deg must be in (0, 90]")
            if key == "refit_multiplier" and value <= 1:
                raise ValueError("refit_multiplier must be greater than 1")
    return result


@dataclass
class _Geometry:
    counts: np.ndarray
    centers: np.ndarray
    scatters: np.ndarray
    normals: np.ndarray
    planarity: np.ndarray
    scattering: np.ndarray
    valid: np.ndarray


def _geometry(points: np.ndarray, groups: np.ndarray, count: int, report) -> _Geometry:
    counts = np.bincount(groups, minlength=count)
    centers = np.zeros((count, 3), dtype=np.float64)
    scatters = np.zeros((count, 3, 3), dtype=np.float64)
    normals = np.zeros((count, 3), dtype=np.float64)
    planarity = np.zeros(count)
    scattering = np.ones(count)
    valid = np.zeros(count, dtype=bool)
    order = np.argsort(groups, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i in range(count):
        report("Fitting superpoints", i, count)
        members = points[order[offsets[i]:offsets[i + 1]]]
        # Work relative to a member to retain precision at large world offsets.
        local = members - members[0]
        mean = local.mean(axis=0)
        centers[i] = members[0] + mean
        centered = local - mean
        scatters[i] = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(scatters[i])
        eigenvalues = np.maximum(eigenvalues, 0)
        scale = eigenvalues[2]
        if counts[i] >= 3 and scale > 0 and eigenvalues[1] > scale * 1e-12:
            normals[i] = eigenvectors[:, 0]
            planarity[i] = (eigenvalues[1] - eigenvalues[0]) / scale
            scattering[i] = eigenvalues[0] / scale
            valid[i] = True
    report("Fitting superpoints", count, count)
    return _Geometry(counts, centers, scatters, normals, planarity, scattering, valid)


def _touching_graph(points, groups, count, radius, report, diagnostics=None):
    graph, stats = build_touching_graph(points, groups, count, radius, report)
    if diagnostics is not None:
        diagnostics.update(stats)
    return graph


def _fit_region(members: set[int], geom: _Geometry) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(members), dtype=np.intp)
    weights = geom.counts[ids]
    anchor = geom.centers[ids[0]]
    local = geom.centers[ids] - anchor
    mean = np.average(local, axis=0, weights=weights)
    delta = local - mean
    scatter = geom.scatters[ids].sum(axis=0) + (delta * weights[:, None]).T @ delta
    _, vectors = np.linalg.eigh(scatter)
    return anchor + mean, vectors[:, 0]


def _components(members: set[int], graph: list[set[int]], check) -> list[set[int]]:
    remaining = set(members)
    components = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        stack = [root]
        while stack:
            check()
            neighbors = graph[stack.pop()] & remaining
            remaining.difference_update(neighbors)
            component.update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(component)
    return components


def segment_planar_superpoints(
    data_or_coords,
    *,
    instance_field_name: str | None = None,
    instance_labels=None,
    selection_indices=None,
    params: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return full-cloud labels and diagnostics without modifying the input.

    Exactly one of ``instance_field_name`` and ``instance_labels`` is required.
    Named fields must have type INSTANCE. Empty inputs/selections return all -1.
    A source ID remains indivisible even if its own member points are disconnected.
    """
    started = perf_counter()

    def check():
        if cancel_requested is not None and cancel_requested():
            raise SuperpointGrowthCancelled("Superpoint region growing cancelled")

    def report(stage, done, total):
        check()
        if progress_callback is not None:
            progress_callback(stage, int(done), int(total))
        check()

    report("Preparing superpoints", 0, 1)
    p = _parameters(params)
    raw = np.asarray(getattr(data_or_coords, "points", data_or_coords))
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError("coords must be a real numeric (N,3) array")
    coords = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("coords must be finite")
    if (instance_field_name is None) == (instance_labels is None):
        raise ValueError("Provide exactly one of instance_field_name and instance_labels")
    if instance_field_name is not None:
        if not hasattr(data_or_coords, "get_fields"):
            raise ValueError("instance_field_name requires PointCloudData")
        fields = data_or_coords.get_fields(names=instance_field_name)
        if not fields or getattr(fields[0].field_type, "name", None) != "INSTANCE":
            raise ValueError(f"Instance field '{instance_field_name}' was not found")
        instance_labels = fields[0].data
    source_labels = np.asarray(instance_labels)
    if source_labels.shape not in {(len(coords),), (len(coords), 1)} or source_labels.dtype.kind not in "iu":
        raise ValueError("instance_labels must be integer labels shaped (N,) or (N,1)")
    source_labels = source_labels.reshape(-1)
    included = source_labels >= 0
    if selection_indices is not None:
        selection = np.asarray(selection_indices)
        if selection.ndim != 1 or (selection.size and selection.dtype.kind not in "iu"):
            raise ValueError("selection_indices must be a one-dimensional integer array")
        if selection.size and ((selection < 0).any() or (selection >= len(coords)).any()):
            raise ValueError("selection_indices are out of bounds")
        selected_ids = np.unique(source_labels[selection.astype(np.intp)])
        included &= np.isin(source_labels, selected_ids[selected_ids >= 0])
    point_indices = np.flatnonzero(included)
    source_ids, groups = np.unique(source_labels[included], return_inverse=True)
    points = coords[point_indices]
    report("Preparing superpoints", 1, 1)
    geom = _geometry(points, groups, len(source_ids), report)
    graph_stats = {}
    graph = _touching_graph(points, groups, len(source_ids), p["min_d"], report, graph_stats)
    assignments = np.full(len(source_ids), -1, dtype=np.int32)
    cosine = np.cos(np.deg2rad(p["alpha_deg"]))
    plane_tolerance = p["epsilon"] * p["epsilon_multiplier"]

    def accepts(i, center, normal):
        if not geom.valid[i]:
            return False
        local = geom.centers[i] - center
        threshold = cosine + (1 - cosine) * np.clip(np.linalg.norm(local) / p["max_dist_from_cent"], 0, 1)
        return (abs(local @ normal) <= plane_tolerance
                and abs(geom.normals[i] @ normal) > threshold)

    def prune(members, center, normal):
        kept = set()
        for i in sorted(members):
            check()
            if accepts(i, center, normal):
                kept.add(i)
        return kept

    seeds = np.lexsort((source_ids, geom.scattering, -geom.planarity))
    attempts = regions = 0
    for seed_number, seed in enumerate(seeds):
        report("Growing regions", seed_number, len(seeds))
        seed = int(seed)
        if assignments[seed] >= 0 or not geom.valid[seed]:
            continue
        if p["enable_seed_gating"] and (
            geom.counts[seed] < p["seed_min_points"]
            or geom.planarity[seed] < p["seed_planarity_min"]
            or geom.scattering[seed] > p["seed_scattering_max"]
        ):
            continue
        attempts += 1
        members, frontier, banned = {seed}, {seed}, set()
        center, normal = geom.centers[seed], geom.normals[seed]
        next_refit = p["first_refit"]
        while frontier:
            check()
            candidates = set()
            for i in sorted(frontier):
                check()
                candidates.update(graph[i])
            candidates.difference_update(members | banned)
            frontier = set()
            for candidate in sorted(candidates):
                check()
                if assignments[candidate] < 0 and accepts(candidate, center, normal):
                    members.add(candidate)
                    frontier.add(candidate)
            if members and len(members) >= next_refit:
                while next_refit <= len(members):
                    next_refit = max(next_refit + 1, int(np.ceil(next_refit * p["refit_multiplier"])))
                center, normal = _fit_region(members, geom)
                kept = prune(members, center, normal)
                banned.update(members - kept)
                members = kept
                frontier.intersection_update(kept)

        # Removal/splitting is monotonic, so stabilization always terminates.
        pending = _components(members, graph, check)
        while pending:
            check()
            component = pending.pop(0)
            if sum(int(geom.counts[i]) for i in component) < p["tau"]:
                continue
            center, normal = _fit_region(component, geom)
            kept = prune(component, center, normal)
            if kept != component:
                pending.extend(_components(kept, graph, check))
                continue
            assignments[list(sorted(component))] = regions
            regions += 1
    report("Growing regions", len(seeds), len(seeds))
    output = np.full(len(coords), -1, dtype=np.int32)
    output[point_indices] = assignments[groups]
    stats = {
        **graph_stats,
        "num_points": len(coords),
        "num_processed_points": len(points),
        "num_superpoints": len(source_ids),
        "num_edges": sum(map(len, graph)) // 2,
        "num_invalid_normals": int((~geom.valid).sum()),
        "num_seed_attempts": attempts,
        "num_regions": regions,
        "num_unassigned": int((output < 0).sum()),
        "elapsed_seconds": perf_counter() - started,
    }
    report("Done", 1, 1)
    return output, stats
