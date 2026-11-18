import numpy as np
import pytest

from geon.algorithms import superpoint_region_growing as rg


def patch(x=0.0, z=0.0, vertical=False):
    a, b = np.meshgrid(np.linspace(0, 0.3, 4), np.linspace(0, 0.3, 4))
    if vertical:
        return np.column_stack((np.full(a.size, x), b.ravel(), a.ravel() + z))
    return np.column_stack((a.ravel() + x, b.ravel(), np.full(a.size, z)))


def grow(parts, ids=None, **params):
    ids = np.arange(len(parts)) if ids is None else ids
    points = np.concatenate(parts)
    source = np.concatenate([np.full(len(p), i, dtype=np.int64) for p, i in zip(parts, ids)])
    labels, stats = rg.segment_planar_superpoints(
        points, instance_labels=source,
        params={"min_d": 0.02, "tau": 3, **params})
    return labels, stats


def test_graph_matches_brute_force_and_exact_threshold():
    rng = np.random.default_rng(12)
    points = np.vstack((rng.random((40, 3)), [[0, 0, 0], [0.25, 0, 0], [0, 0, 0]]))
    groups = np.arange(len(points)) % 7
    for radius in [0.0, 0.25, 0.5, 2.0]:
        graph = rg._touching_graph(points, groups, 7, radius, lambda *args: None)
        expected = [set() for _ in range(7)]
        for a in range(len(points)):
            for b in range(a):
                if groups[a] != groups[b] and np.linalg.norm(points[a] - points[b]) <= radius:
                    expected[groups[a]].add(groups[b])
                    expected[groups[b]].add(groups[a])
        assert graph == expected
    graph = rg._touching_graph(np.array([[0., 0, 0], [.25, 0, 0]]), np.array([0, 1]), 2, .25, lambda *a: None)
    assert graph == [{1}, {0}]


def test_touching_boundaries_merge_distant_centroids_and_sparse_ids():
    labels, stats = grow([patch(), patch(.31), patch(.62)], [0, 800, 2**40])
    assert np.all(labels == 0)
    assert labels.dtype == np.int32
    assert stats["num_superpoints"] == 3
    assert stats["num_edges"] == 2
    assert stats["num_regions"] == 1


@pytest.mark.parametrize("other,params", [
    (patch(2), {}),
    (patch(.31, vertical=True), {}),
    (patch(z=.1), {"min_d": .2, "epsilon": .01, "epsilon_multiplier": 1}),
])
def test_disconnected_angular_and_offset_separation(other, params):
    labels, stats = grow([patch(), other], **params)
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[:16])) == len(np.unique(labels[16:])) == 1
    assert stats["num_regions"] == 2


def test_tau_counts_points_and_failed_regions_release_members():
    labels, _ = grow([patch(), patch(.31)], tau=32)
    assert np.all(labels == 0)
    labels, stats = grow([patch(), patch(.31)], tau=33)
    assert np.all(labels == -1)
    assert stats["num_seed_attempts"] == 2


def test_selection_expands_to_whole_instances_and_ignores_negative_ids():
    points = np.concatenate([patch(), patch(.31), patch(.62), patch(.93)])
    source = np.repeat([200, 2, -1, 500], 16)
    original_points, original_source = points.copy(), source.copy()
    labels, stats = rg.segment_planar_superpoints(
        points, instance_labels=source[:, None], selection_indices=[1, 17, 33],
        params={"min_d": .02, "tau": 3})
    np.testing.assert_array_equal(labels, np.repeat([0, 0, -1, -1], 16))
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(source, original_source)
    assert stats["num_processed_points"] == 32
    assert stats["num_unassigned"] == 32


def test_cached_fit_matches_all_points_at_large_world_offset():
    parts = [patch(), np.repeat(patch(.31, z=.01), 7, axis=0)]
    points = np.concatenate(parts) + [1e9, -2e9, 3e9]
    groups = np.repeat([0, 1], [len(p) for p in parts])
    geom = rg._geometry(points, groups, 2, lambda *args: None)
    center, normal = rg._fit_region({0, 1}, geom)
    local = points - points[0]
    mean = local.mean(axis=0)
    centered = local - mean
    _, vectors = np.linalg.eigh(centered.T @ centered)
    np.testing.assert_allclose(center, points[0] + mean, rtol=0, atol=5e-7)
    assert abs(normal @ vectors[:, 0]) > 1 - 1e-10


def test_degenerate_normals_empty_inputs_and_empty_selection():
    points = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 0, 0]])
    labels, stats = rg.segment_planar_superpoints(points, instance_labels=[0, 0, 0, 1, 1])
    assert np.all(labels == -1)
    assert stats["num_invalid_normals"] == 2
    labels, stats = rg.segment_planar_superpoints(np.empty((0, 3)), instance_labels=np.empty(0, dtype=int))
    assert labels.size == stats["num_regions"] == stats["num_superpoints"] == 0
    labels, stats = rg.segment_planar_superpoints(patch(), instance_labels=np.zeros(16, dtype=int), selection_indices=[])
    assert np.all(labels == -1)
    assert stats["num_superpoints"] == 0


def test_seed_gating_only_applies_to_seed_and_results_are_deterministic():
    small = patch(.31)[[0, 3, 12, 15]]
    results = [grow([patch(), small], first_refit=2) for _ in range(2)]
    assert np.all(results[0][0] == 0)
    np.testing.assert_array_equal(results[0][0], results[1][0])
    labels, _ = grow([small], enable_seed_gating=True)
    assert np.all(labels == -1)
    labels, _ = grow([small], enable_seed_gating=False)
    assert np.all(labels == 0)


def test_refit_prunes_bridge_and_splits_remaining_graph(monkeypatch):
    # Seed z, bridge +20 degrees, large last patch -20 degrees. A combined
    # refit follows the large patch, ejecting the bridge and disconnecting ends.
    normals = np.array([[0., 0, 1], [0, np.sin(np.pi/9), np.cos(np.pi/9)],
                        [0, -np.sin(np.pi/9), np.cos(np.pi/9)]])
    counts = np.array([16, 16, 16000])
    scatters = np.array([(np.eye(3) - np.outer(n, n)) * count for n, count in zip(normals, counts)])
    geom = rg._Geometry(counts, np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]]),
                        scatters, normals, np.array([1., .8, .7]), np.zeros(3), np.ones(3, dtype=bool))
    monkeypatch.setattr(rg, "_geometry", lambda *args: geom)
    monkeypatch.setattr(rg, "_touching_graph", lambda *args: [{1}, {0, 2}, {1}])
    labels, stats = rg.segment_planar_superpoints(
        np.zeros((3, 3)), instance_labels=np.arange(3),
        params={"tau": 3, "first_refit": 3, "alpha_deg": 29})
    assert labels[0] >= 0 and labels[2] >= 0 and labels[0] != labels[2]
    assert stats["num_regions"] == 3  # Bridge is subsequently tried on its own.


@pytest.mark.parametrize("stage", ["Preparing superpoints", "Fitting superpoints", "Building touching graph", "Growing regions"])
def test_cooperative_cancellation(stage):
    cancelled = False
    def progress(current, done, total):
        nonlocal cancelled
        if current == stage:
            cancelled = True
    with pytest.raises(rg.SuperpointGrowthCancelled):
        rg.segment_planar_superpoints(patch(), instance_labels=np.zeros(16, dtype=int),
                                      progress_callback=progress, cancel_requested=lambda: cancelled)


@pytest.mark.parametrize("kwargs", [
    {"instance_labels": np.zeros(16, dtype=float)},
    {"instance_labels": np.zeros((16, 2), dtype=int)},
    {"selection_indices": [16]}, {"selection_indices": [-1]}, {"selection_indices": [1.5]},
    {"params": {"min_d": -1}}, {"params": {"epsilon": float("nan")}},
    {"params": {"tau": 2.5}}, {"params": {"refit_multiplier": 1}},
    {"params": {"alpha_deg": 91}}, {"params": {"enable_seed_gating": 1}},
    {"params": {"seed_planarity_min": 1.1}}, {"params": {"unknown": 1}},
    {"instance_field_name": "a"},
])
def test_validation(kwargs):
    args = {"instance_labels": np.zeros(16, dtype=int), **kwargs}
    with pytest.raises(ValueError):
        rg.segment_planar_superpoints(patch(), **args)


def test_invalid_coordinates_and_missing_labels():
    points = patch()
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rg.segment_planar_superpoints(points, instance_labels=np.zeros(16, dtype=int))
    with pytest.raises(ValueError, match="exactly one"):
        rg.segment_planar_superpoints(patch())


def test_field_adapter():
    from geon.data.pointcloud import PointCloudData, FieldType
    data = PointCloudData(patch())
    data.add_field("instances", np.zeros((16, 1), dtype=np.int32), FieldType.INSTANCE)
    data.add_field("scalar", np.zeros((16, 1)), FieldType.SCALAR)
    labels, _ = rg.segment_planar_superpoints(data, instance_field_name="instances", params={"tau": 3})
    assert np.all(labels == 0)
    for name in ["scalar", "missing"]:
        with pytest.raises(ValueError, match="Instance field"):
            rg.segment_planar_superpoints(data, instance_field_name=name)
