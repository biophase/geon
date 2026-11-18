"""Exact, memory-aware contact graphs for compact integer point labels."""
from collections import OrderedDict
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

TEMP_BYTES = 256 * 1024**2
TREE_CACHE_BYTES = 128 * 1024**2


def _insert_keys(graph, keys):
    for key in keys:
        a, b = int(key) >> 32, int(key) & 0xffffffff
        graph[a].add(b)
        graph[b].add(a)


def _edge_keys(a, b):
    a, b = np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)
    return np.unique((np.minimum(a, b) << np.uint64(32)) | np.maximum(a, b))


def _zero_graph(points, groups, graph, report):
    # Sort indices, not an N-by-3 coordinate copy. Only duplicate runs need
    # Python work; the overwhelmingly common singleton runs are skipped.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    pending = None

    def insert_run(start, end):
        ids = np.unique(groups[order[start:end + 1]])
        for pos, a in enumerate(ids):
            if pos % 256 == 0:
                report("Building touching graph", start, len(points))
            for b in ids[pos + 1:]:
                graph[int(a)].add(int(b))
                graph[int(b)].add(int(a))

    for start in range(0, len(points) - 1, 131072):
        report("Grouping identical coordinates", start, len(points))
        end = min(start + 131072, len(points) - 1)
        equal = np.all(points[order[start:end]] == points[order[start + 1:end + 1]], axis=1)
        changes = np.flatnonzero(np.diff(np.r_[False, equal, False])) + start
        for lo, hi in changes.reshape(-1, 2):
            if pending is not None and pending[1] == lo:
                pending = (pending[0], hi)
            else:
                if pending is not None:
                    insert_run(*pending)
                pending = (lo, hi)
        if pending is not None and pending[1] < end:
            insert_run(*pending)
            pending = None
    if pending is not None:
        insert_run(*pending)


def _vector_graph(tree, points, groups, radius, search_radius, graph, report):
    pairs = tree.query_pairs(search_radius, eps=0, output_type="ndarray")
    report("Deduplicating touching edges", 0, len(pairs))
    edge_blocks = []
    # Work arrays stay bounded even when the pair array approaches its budget.
    for start in range(0, len(pairs), 131072):
        report("Deduplicating touching edges", start, len(pairs))
        block = pairs[start:start + 131072]
        a, b = groups[block[:, 0]], groups[block[:, 1]]
        block = block[a != b]
        distance = np.linalg.norm(points[block[:, 0]] - points[block[:, 1]], axis=1)
        block = block[distance <= radius]
        edge_blocks.append(_edge_keys(groups[block[:, 0]], groups[block[:, 1]]))
    # A label pair can occur in many point batches. Deduplicate globally before
    # creating Python sets so each final edge is inserted only once.
    keys = np.unique(np.concatenate(edge_blocks)) if edge_blocks else np.empty(0, dtype=np.uint64)
    del pairs, edge_blocks
    for start in range(0, len(keys), 65536):
        report("Assembling touching graph", start, len(keys))
        _insert_keys(graph, keys[start:start + 65536])


def _direct_graph(points, groups, count, radius, guard, graph, report):
    order = np.argsort(groups, kind="stable")
    counts = np.bincount(groups, minlength=count)
    offsets = np.r_[0, np.cumsum(counts)]
    occupied = np.flatnonzero(counts)
    low, high = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    # Grouped coordinates would cost another 240 MB at ten million points.
    # Reduce bounded blocks instead and gather coordinates only for local trees.
    for start in range(0, len(points), 131072):
        report("Bounding superpoints", start, len(points))
        ids = groups[start:start + 131072]
        np.minimum.at(low, ids, points[start:start + 131072])
        np.maximum.at(high, ids, points[start:start + 131072])
    low[counts == 0] = high[counts == 0] = 0
    centers = low + (high - low) * .5
    radii = np.linalg.norm((high - low) * .5, axis=1)
    exponents = np.frexp(radii[occupied])[1]
    buckets = []
    for exponent in np.unique(exponents):
        ids = occupied[exponents == exponent]
        buckets.append((ids, cKDTree(centers[ids]), float(radii[ids].max())))
    cache, cache_bytes = OrderedDict(), 0
    candidate_count = 0

    def point_tree(label):
        nonlocal cache_bytes
        if label in cache:
            cache.move_to_end(label)
            return cache[label][0]
        tree = cKDTree(points[order[offsets[label]:offsets[label + 1]]])
        estimate = int(counts[label]) * 64 + 1024
        while cache and cache_bytes + estimate > TREE_CACHE_BYTES:
            _, (_, old_bytes) = cache.popitem(last=False)
            cache_bytes -= old_bytes
        if estimate <= TREE_CACHE_BYTES:
            cache[label] = tree, estimate
            cache_bytes += estimate
        return tree

    for number, a in enumerate(occupied):
        report("Testing superpoint contacts", number, len(occupied))
        for ids, tree, max_radius in buckets:
            bound = np.nextafter(radii[a] + max_radius + radius + guard, np.inf)
            size = tree.query_ball_point(centers[a], bound, return_length=True)
            if size <= 65536:
                candidates = ids[tree.query_ball_point(centers[a], bound)]
                blocks = (candidates,)
            else:
                # Avoid an arbitrarily large Python neighbor list for huge boxes.
                blocks = (ids[k:k + 8192] for k in range(0, len(ids), 8192))
            for block in blocks:
                report("Testing superpoint contacts", number, len(occupied))
                candidates = block[block > a]
                delta = np.maximum(0, np.maximum(low[candidates] - high[a], low[a] - high[candidates]))
                candidates = candidates[np.linalg.norm(delta, axis=1) <= radius + guard]
                for b in candidates:
                    candidate_count += 1
                    small, big = (a, b) if counts[a] <= counts[b] else (b, a)
                    target = point_tree(int(big))
                    for start in range(int(offsets[small]), int(offsets[small + 1]), 2048):
                        report("Testing superpoint contacts", number, len(occupied))
                        query = points[order[start:min(start + 2048, offsets[small + 1])]]
                        gap = np.maximum(0, np.maximum(low[big] - query, query - high[big]))
                        query = query[np.linalg.norm(gap, axis=1) <= radius + guard]
                        if not len(query):
                            continue
                        distances, nearest = target.query(query, k=1, eps=0,
                            distance_upper_bound=np.nextafter(radius + guard, np.inf))
                        valid = np.isfinite(distances)
                        if valid.any() and np.any(np.linalg.norm(
                                query[valid] - target.data[nearest[valid]], axis=1) <= radius):
                            graph[int(a)].add(int(b))
                            graph[int(b)].add(int(a))
                            break
    return candidate_count


def build_touching_graph(points, groups, count, radius, report, *, backend="auto", memory_bytes=TEMP_BYTES):
    """Return adjacency and diagnostics. Forced backends are useful for parity tests.

    Temporary contact work is budgeted separately from coordinates, spatial
    indices and the final graph. Compiled tree calls cannot be interrupted;
    cancellation is checked between those calls and between processing batches.
    """
    started = perf_counter()
    if count >= 2**32:
        raise ValueError("Touching graphs support fewer than 2**32 superpoints")
    if backend not in {"auto", "native", "vector", "direct"}:
        raise ValueError("Unknown touching graph backend")
    native = None
    if backend in {"auto", "native"}:
        try:
            from geon._native.touching_graph import edges as native
        except ImportError:
            if backend == "native":
                raise
    graph = [set() for _ in range(count)]
    report("Building touching graph", 0, len(points))
    candidates = pair_count = 0
    exact_count = True
    if not len(points):
        chosen = "empty"
    elif radius == 0:
        chosen = "exact_coordinates"
        _zero_graph(points, groups, graph, report)
    elif native is not None:
        keys = native(points, groups, radius, report)
        _insert_keys(graph, keys)
        chosen = "native"
        exact_count = False
    else:
        # A conservative search guard protects broad-phase bounds at large
        # world coordinates; the final test always uses the requested radius.
        guard = 16 * np.finfo(float).eps * max(1., float(np.abs(points.min(axis=0)).max()),
                                             float(np.abs(points.max(axis=0)).max()))
        search_radius = np.nextafter(radius + guard, np.inf)
        tree = cKDTree(points) if backend != "direct" else None
        if tree is None:
            exact_count = False
        if tree is not None:
            report("Counting point contacts", 0, 1)
            limit = min(32 * len(points), memory_bytes // 128)
            # A partial directed count is a rigorous lower bound on the total.
            # Dense clouds can be rejected without counting billions of contacts.
            sample = np.linspace(0, len(points)-1, min(len(points), 65536), dtype=np.intp)
            directed = 0
            for start in range(0, len(sample), 8192):
                report("Counting point contacts", 0, 1)
                rows = sample[start:start+8192]
                directed += int(tree.query_ball_point(points[rows], search_radius, return_length=True).sum()) - len(rows)
                if directed > 2*limit:
                    break
            if directed > 2*limit:
                pair_count, exact_count = (directed+1)//2, False
            else:
                pair_count = (int(tree.count_neighbors(tree, search_radius)) - len(points)) // 2
            report("Counting point contacts", 1, 1)
        chosen = backend
        if backend == "auto":
            chosen = "vector" if pair_count <= 32 * len(points) and pair_count * 128 <= memory_bytes else "direct"
        if chosen == "vector":
            if pair_count * 128 > memory_bytes:
                raise ValueError("Point-pair work exceeds the touching graph memory budget")
            _vector_graph(tree, points, groups, radius, search_radius, graph, report)
        else:
            del tree
            candidates = _direct_graph(points, groups, count, radius, guard, graph, report)
    report("Building touching graph", len(points), len(points))
    return graph, {"graph_backend": chosen, "graph_seconds": perf_counter() - started,
                   "graph_point_pairs": pair_count, "graph_point_pairs_exact": exact_count,
                   "graph_candidate_pairs": candidates}
