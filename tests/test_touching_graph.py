import numpy as np
import pytest

from geon.algorithms.touching_graph import build_touching_graph


def brute(points, labels, count, radius):
    graph = [set() for _ in range(count)]
    for i in range(len(points)):
        for j in range(i):
            if labels[i] != labels[j] and np.linalg.norm(points[i] - points[j]) <= radius:
                graph[labels[i]].add(int(labels[j]))
                graph[labels[j]].add(int(labels[i]))
    return graph


@pytest.mark.parametrize("backend", ["vector", "direct", "auto", "native"])
@pytest.mark.parametrize("radius", [0., .25, .8])
def test_exact_backend_parity(backend, radius):
    if backend == "native": pytest.importorskip('geon._native.touching_graph')
    rng = np.random.default_rng(82)
    points = np.vstack([rng.random((100, 3)), [[0, 0, 0], [.25, 0, 0], [0, 0, 0]]])
    labels = np.arange(len(points)) % 13
    graph, stats = build_touching_graph(points, labels, 15, radius, lambda *args: None, backend=backend)
    assert graph == brute(points, labels, 15, radius)
    assert stats["graph_seconds"] >= 0


def test_radius_buckets_elongated_boxes_and_large_offsets():
    points = np.array([[0., 0, 0], [20, 0, 0], [10, .1, 0], [20, .25, 0],
                       [20, .5, 0], [21, .25, 0], [1, 1, 0]]) + 1e9
    labels = np.array([0, 0, 1, 2, 3, 4, 5])
    graph, _ = build_touching_graph(points, labels, 6, .25, lambda *args: None, backend="direct")
    assert graph == brute(points, labels, 6, .25)
    assert 1 not in graph[0]  # Overlapping AABBs are only a broad-phase test.


def test_auto_memory_budget_and_dense_selection():
    points = np.random.default_rng(1).random((100, 3))
    labels = np.arange(100) % 5
    graph, stats = build_touching_graph(points, labels, 5, 2., lambda *args: None, memory_bytes=1)
    assert stats["graph_backend"] in {"direct", "native"}
    assert not stats["graph_point_pairs_exact"]
    assert graph == brute(points, labels, 5, 2.)
    with pytest.raises(ValueError, match="memory budget"):
        build_touching_graph(points, labels, 5, 2., lambda *args: None, backend="vector", memory_bytes=1)


@pytest.mark.parametrize("backend", ["vector", "direct", "native"])
def test_cancellation_after_search_begins(backend):
    if backend == "native": pytest.importorskip('geon._native.touching_graph')
    stages = []
    def report(stage, done, total):
        stages.append(stage)
        if stage in {"Deduplicating touching edges", "Testing superpoint contacts", "Streaming superpoint contacts"}:
            raise InterruptedError("cancelled")
    points = np.random.default_rng(1).random((100, 3))
    with pytest.raises(InterruptedError):
        build_touching_graph(points, np.arange(100) % 5, 5, .1, report, backend=backend)


def test_identical_coordinates_across_batch_boundary():
    n = 131080
    points = np.column_stack((np.arange(n, dtype=float), np.zeros((n, 2))))
    points[131070:] = points[131070]
    labels = np.zeros(n, dtype=int)
    labels[131075:] = 1
    graph, _ = build_touching_graph(points, labels, 2, 0., lambda *args: None)
    assert graph == [{1}, {0}]


def test_native_inclusive_non_axis_aligned_threshold():
    pytest.importorskip('geon._native.touching_graph')
    rng=np.random.default_rng(4)
    for offset in (0.,1e9):
        for _ in range(100):
            points=rng.random((2,3))+offset
            # Match the vector backend's per-row norm (a 1D BLAS dot can
            # round differently by one ULP at this exact boundary).
            radius=np.linalg.norm((points[0]-points[1])[None,:],axis=1)[0]
            graph,_=build_touching_graph(points,np.array([0,1]),2,radius,lambda *a:None,backend='native')
            assert graph==[{1},{0}]
            graph,_=build_touching_graph(points,np.array([0,1]),2,np.nextafter(radius,0.),
                                         lambda *a:None,backend='native')
            assert graph==[set(),set()]
