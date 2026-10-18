"""Histogram calculation has no rendering or HDF5 dependencies."""
import numpy as np
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal


def calculate_histogram(values, mask=None, limits=None, categorical=False):
    visible = [v if mask is None else v[mask] for v in values]
    visible = [v[np.isfinite(v)] for v in visible]
    if categorical:
        labels, counts = np.unique(visible[0], return_counts=True)
        return labels, [counts]
    nonempty = [v for v in visible if v.size]
    if not nonempty:
        return np.array([]), []
    low, high = limits if limits is not None else (min(v.min() for v in nonempty), max(v.max() for v in nonempty))
    if low == high:
        low, high = low-.5, high+.5
    edges = np.linspace(low, high, 257)
    return edges, [np.histogram(v, bins=edges)[0] for v in visible]


class HistogramResult(QObject):
    ready = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)


class HistogramJob(QRunnable):
    def __init__(self, generation, values, mask, limits, categorical):
        super().__init__()
        self.result = HistogramResult()
        self.generation = generation
        self.args = values, mask, limits, categorical

    def run(self):
        try:
            edges, counts = calculate_histogram(*self.args)
            self.result.ready.emit(self.generation, edges, counts)
        except Exception as exc:
            self.result.failed.emit(self.generation, str(exc))
