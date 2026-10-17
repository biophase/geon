"""TEMP_VISIBILITY_PROFILE: remove this module after investigation.

Temporary CPU wall-clock profiling for lasso/visibility operations.

Enabled by default; GEON_PROFILE_VISIBILITY=0 disables it. Nested times are
inclusive, and Render timings do not separately measure GPU execution.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from time import perf_counter

_ENABLED = os.environ.get("GEON_PROFILE_VISIBILITY", "1") != "0"
_trace = ContextVar("visibility_timing", default=None)


@contextmanager
def timing(label, *, root=False):
    trace = _trace.get()
    if not _ENABLED or (trace is None and not root):
        yield
        return
    owner = trace is None
    if owner:
        trace = {"rows": [], "depth": 0}
        token = _trace.set(trace)
    depth = trace["depth"]
    row = [depth, label, 0.0]
    trace["rows"].append(row)
    trace["depth"] += 1
    start = perf_counter()
    try:
        yield
    finally:
        row[2] = perf_counter() - start
        trace["depth"] -= 1
        if owner:
            _trace.reset(token)
            lines = ["[visibility-profile] CPU wall times (nested/inclusive):"]
            lines.extend(f"  {'  ' * d}{name}" + (f": {elapsed * 1000:.2f} ms" if elapsed is not None else "")
                         for d, name, elapsed in trace["rows"])
            print("\n".join(lines), flush=True)


def timed(label, *, root=False):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with timing(label, root=root):
                return func(*args, **kwargs)
        return wrapped
    return decorate


def timing_note(message):
    trace = _trace.get()
    if trace is not None:
        trace["rows"].append([trace["depth"], message, None])


def profile_render(render_window, renderer, scene):
    """Observe actual upstream execution without forcing extra mapper updates.

    TEMP_VISIBILITY_PROFILE: remove this helper and restore Render() at caller.
    The remainder includes unobserved VTK work, uploads, drawing and waits;
    it is deliberately not labelled GPU time.
    """
    if _trace.get() is None:
        render_window.Render()
        return

    mappers = []
    actors = renderer.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        if actor.GetVisibility() and actor.GetMapper() is not None:
            mappers.append(("actor", actor.GetMapper()))
    # LOD selection happens inside Render; observe both possible pipelines.
    if scene is not None:
        for layer in scene.layers.values():
            for attr in ("_mapper_fine", "_mapper_coarse"):
                mapper = getattr(layer, attr, None)
                if mapper is not None:
                    mappers.insert(0, (attr.removeprefix("_mapper_"), mapper))

    observers, spans, seen = [], [], set()

    def observe(algorithm, label):
        key = algorithm.GetAddressAsString("")
        if key in seen:
            return
        seen.add(key)
        starts = []

        def start(*_):
            starts.append(perf_counter())

        def end(*_):
            if starts:
                spans.append((starts.pop(), perf_counter(), label))

        observers.append((algorithm, algorithm.AddObserver("StartEvent", start)))
        observers.append((algorithm, algorithm.AddObserver("EndEvent", end)))
        for port in range(algorithm.GetNumberOfInputPorts()):
            for connection in range(algorithm.GetNumberOfInputConnections(port)):
                upstream = algorithm.GetInputAlgorithm(port, connection)
                if upstream is not None:
                    observe(upstream, f"{label} <- {upstream.GetClassName()}")

    try:
        for name, mapper in mappers:
            # Mapper events may encompass rendering, so observe only inputs.
            for port in range(mapper.GetNumberOfInputPorts()):
                for connection in range(mapper.GetNumberOfInputConnections(port)):
                    algorithm = mapper.GetInputAlgorithm(port, connection)
                    if algorithm is not None:
                        observe(algorithm, f"{name}: {algorithm.GetClassName()}")
        start = perf_counter()
        render_window.Render()
        elapsed = perf_counter() - start
        trace = _trace.get()
        for begin, end, label in sorted(spans):
            trace["rows"].append([trace["depth"], f"Pipeline execution [{label}]", end - begin])
        # Union intervals so nested algorithm events are not double-counted.
        covered = 0.0
        last_end = float("-inf")
        for begin, end, _ in sorted(spans):
            covered += max(0.0, end - max(begin, last_end))
            last_end = max(last_end, end)
        trace["rows"].append([trace["depth"], "Observed pipeline execution (union)", covered])
        trace["rows"].append([trace["depth"], "Render remainder (other VTK work + uploads/draw/wait)", max(0.0, elapsed - covered)])
        if scene is not None:
            for layer in scene.layers.values():
                actor = getattr(layer, "_main_actor", None)
                fine = getattr(layer, "_mapper_fine", None)
                if actor is not None and fine is not None and actor.GetVisibility():
                    mapper = actor.GetMapper()
                    data = mapper.GetInput()
                    count = data.GetNumberOfPoints() if data is not None else 0
                    timing_note(f"Rendered point cloud: {'fine' if mapper is fine else 'coarse'} LOD; {count:,} points")
    finally:
        for algorithm, tag in observers:
            algorithm.RemoveObserver(tag)
