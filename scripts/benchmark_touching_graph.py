"""Run each case in a fresh process to measure graph time and peak process RSS.

Usage: python scripts/benchmark_touching_graph.py --points 1000000 --density sparse
Requires psutil for memory sampling; it is a benchmark-only dependency.
"""
import argparse
import json
import threading
import time

import numpy as np
import psutil

from geon.algorithms.touching_graph import build_touching_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1000000)
    parser.add_argument("--density", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--backend", choices=["auto", "native", "direct", "vector"], default="auto")
    args = parser.parse_args()
    per_group = 25 if args.density == "sparse" else 400
    count = (args.points + per_group - 1) // per_group
    width = int(np.ceil(np.sqrt(count)))
    ids = np.arange(count)
    origins = np.column_stack((ids % width, ids // width, np.zeros(count)))
    if args.density == "sparse":
        u, v = np.meshgrid(np.arange(5) * .01, np.arange(5) * .01)
        patch = np.column_stack((u.ravel(), v.ravel(), np.zeros(25)))
        points = (origins[:, None, :] * .05 + patch).reshape(-1, 3)[:args.points]
        radius = .010001
    else:
        points = np.repeat(origins * .02, per_group, axis=0)[:args.points]
        rng = np.random.default_rng(42)
        for start in range(0, len(points), 100000):
            stop = min(start + 100000, len(points))
            points[start:stop] += rng.uniform(0, .02, (stop - start, 3))
        radius = .015
    labels = np.repeat(ids, per_group)[:args.points]
    process = psutil.Process()
    peak = [process.memory_info().rss]
    baseline = peak[0]
    stop = threading.Event()
    def sample():
        while not stop.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    graph, stats = build_touching_graph(points, labels, count, radius, lambda *args: None, backend=args.backend)
    stop.set()
    monitor.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    print(json.dumps(dict(stats, density=args.density, points=len(points), superpoints=count,
        edges=sum(map(len, graph)) // 2, elapsed_seconds=time.perf_counter() - started,
        peak_rss_mib=peak[0] / 1024**2, baseline_rss_mib=baseline / 1024**2)))


if __name__ == "__main__":
    main()
